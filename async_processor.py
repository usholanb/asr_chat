from typing import Generator, Iterator

import openai
import torch.cuda
from utils.helpers import get_secret, append_to_file, LimitedSizeDict, \
    get_file_content, return_default
import config
import os
import paths
from utils.helpers import catch_time, get_transcribe_wav_file
import asyncio
import utils.constants as const
from bullmq import Worker, Job, Queue
from bullmq.types import WorkerOptions
from sync_modules.bullmq_objects import Recording, JobResult
from vellum.client import Vellum
from vellum.types import GenerateRequest
from translate import Translator
import openai


openai.api_key = get_secret('OPENAI_API_KEY')
device = 'gpu' if torch.cuda.is_available() else 'cpu'

if config.STRAIGHT_TO_LANG:
    def transcribe_wav_file(audio_file_path) -> str:
        with open(audio_file_path, 'rb') as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file, language=config.TO_LANG)
        return transcription['text']
else:
    # transcribe_wav_file = get_transcribe_wav_file(
    #     config.WHISPER_MODEL_NAME, device)
    def transcribe_wav_file(audio_file_path) -> str:
        with open(audio_file_path, 'rb') as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file, language=config.FROM_LANG)
        return transcription['text']

client = Vellum(api_key=get_secret("VELLUM_API"))
translator = Translator(from_lang=config.FROM_LANG, to_lang=config.TO_LANG)


def call_llm(text: str, tr_scrit_path: str, llm_translation_path: str,
             project_context: str) -> str:
    transcription = get_file_content(tr_scrit_path,
                                     last=config.LAST_CONTEXT_SENTENCES)
    llm_translation = get_file_content(llm_translation_path,
                                       last=config.LAST_CONTEXT_SENTENCES)
    deployment_name = f"translator-with-context-{config.TO_LANG}"
    if config.TO_LANG == 'en':
        deployment_name = deployment_name.split('-en', -1)[0]
    return client.generate(
        deployment_name=deployment_name,
        requests=[
            GenerateRequest(
                input_values={"history_transcribed": transcription,
                              "history_translation": llm_translation,
                              "input_transcribed": text,
                              "project_context": project_context}
            )
        ]
    ).text.strip('"')


def get_llm_generator(text: str, tr_scrit_path: str, llm_translation_path: str,
             project_context: str) -> Iterator:
    transcription = get_file_content(tr_scrit_path,
                                     last=config.LAST_CONTEXT_SENTENCES)
    llm_translation = get_file_content(llm_translation_path,
                                       last=config.LAST_CONTEXT_SENTENCES)

    return client.generate_stream(
        deployment_name=f"translator-with-context-{config.FROM_LANG}-{config.TO_LANG}",
        requests=[
            GenerateRequest(
                input_values={"history_transcribed": transcription,
                              "history_translation": llm_translation,
                              "input_transcribed": text,
                              "project_context": project_context}
            )
        ]
    )

def get_next_index(path: str) -> int:
    return int(path.split('.wav')[0].split('_')[-1])


async def processor(job: Job, job_token):
    ############################## INITIALIZE #################################
    req = Recording.parse_obj(job.data)
    print(f"started processing {req.path}")
    llm_translation_path = f"{paths.LLM_TRANSLATIONS_RESULTS_DIR}/" \
                           f"{req.session_id}.txt"
    tr_scrit_path = f"{paths.TRANSCRIPTIONS_RESULTS_DIR}/{req.session_id}.txt"
    translation_path = f"{paths.TRANSLATIONS_RESULTS_DIR}/{req.session_id}.txt"
    next_index = get_next_index(req.path)

    ############################## TRANSCRIBE #################################
    with catch_time(function_name='transcribe_wav_file(latest_recording)'):
        transcription = return_default(transcribe_wav_file(req.path), "")
    print(f"transcription, text: {transcription}")

    ######################## GOOGLE TRANSLATE #################################
    translated_text = translator.translate(transcription)
    print(f"translated_text: {translated_text}")

    append_to_file(filepath=translation_path, text=translated_text)
    append_to_file(filepath=tr_scrit_path, text=transcription)
    ######################## PROMPT AND LLM TRANSLATE #####################
    if transcription is not None and transcription != '':
        # llm_translated_text = call_llm(transcription, tr_scrit_path,
        #                                llm_translation_path, req.project_context)
        # print(f"completion: {llm_translated_text}")
        #
        # append_to_file(filepath=llm_translation_path,
        #                text=llm_translated_text)
        #
        # full_text = get_file_content(filepath=llm_translation_path)
        # print(f"Full text job {next_index} (number of words = "
        #       f"{len(full_text.split())}):\n{full_text}")
        for llm_output in get_llm_generator(
                transcription, tr_scrit_path,  llm_translation_path, req.project_context):
            llm_translated_text = llm_output.dict()['delta']['data']['completion']['text']
            append_to_file(filepath=llm_translation_path,
                           text=llm_translated_text, sep='')
        full_text = get_file_content(filepath=llm_translation_path)
        print(f"Full text job {next_index} (number of words = "
              f"{len(full_text.split())}):\n{full_text}")
    else:
        full_text = ""

    return JobResult.success(req, full_text).json()


async def straingt_to_lang_processor(job: Job, job_token):

    ############################## INITIALIZE #################################
    req = Recording.parse_obj(job.data)
    print(f"started processing {req.path}")
    llm_translation_path = f"{paths.LLM_TRANSLATIONS_RESULTS_DIR}/" \
                           f"{req.session_id}.txt"
    next_index = get_next_index(req.path)
    ############################## TRANSCRIBE #################################

    llm_translated_text = return_default(transcribe_wav_file(req.path), "")
    print(f"transcription, text: {llm_translated_text}")

    append_to_file(filepath=llm_translation_path,
                   text=llm_translated_text)

    full_text = get_file_content(filepath=llm_translation_path)
    print(f"Full text job {next_index} (number of words = "
          f"{len(full_text.split())}):\n{full_text}")
    return JobResult.success(req, full_text).json()


async def main():
    if config.STRAIGHT_TO_LANG:
        worker_function = straingt_to_lang_processor
    else:
        worker_function = processor

    redis_opts = os.environ.get("REDISCLOUD_URL", "redis://localhost:6379")
    opts: WorkerOptions = {
        "concurrency": 1, "lockDuration": 100 * 1000,
        "autorun": False, "connection": redis_opts
    }
    await asyncio.gather(
        *[Worker(const.RECORDINGS_QUEUE, worker_function, opts).run()
          for _ in range(const.MAX_WORKERS_TRANSCRIBE)])


if __name__ == '__main__':
    import redis
    r = redis.Redis()
    r.flushdb()
    # print(transcribe_wav_file("/Users/uan/PycharmProjects/asr_chat/results/recordings/123_1_1.wav"))
    asyncio.run(main())