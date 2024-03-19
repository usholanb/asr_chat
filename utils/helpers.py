import os
import pickle
import re
from typing import Optional, Callable, Union
from collections import OrderedDict
import whisper
from dotenv import load_dotenv
from time import perf_counter
from contextlib import contextmanager
import config
import paths
from pytube import YouTube
from pydub import AudioSegment


load_dotenv()


def get_secret(name):
    if name in os.environ:
        return os.environ[name]
    return os.getenv(name)


@contextmanager
def catch_time(function_name: str = '', debug=config.DEBUG) -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    if not debug:
        print(f'function_name: {function_name}, Time: {perf_counter() - start:.3f} seconds')


def transcribe_wav_file(wav_file, model_name):
    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"{wav_file} not found.")

    model = whisper.load_model(model_name)

    transcript = ""
    for i in range(0, len(audio), 30*sample_rate):
        result = ''
        chunk = audio[i:i+30*sample_rate]
        with catch_time(function_name='whisper.pad_or_trim'):
            chunk = whisper.pad_or_trim(chunk)
        with catch_time(function_name='whisper.log_mel_spectrogram'):
            mel = whisper.log_mel_spectrogram(chunk).to(model.device)

        options = whisper.DecodingOptions(language=config.FROM_LANG, fp16=False)
        with catch_time(function_name='whisper.decode'):
            result = whisper.decode(model, mel, options).text
        transcript += result + " "
        # if result.no_speech_prob < 0.25:

        # else:
        #     print("No speech detected in the recording.")
    return transcript


def get_transcribe_wav_file(model_name: str, device: str) -> Callable:
    print(f"started loading model {model_name}")
    model = whisper.load_model(model_name, device=device)
    print(f"loaded model {model_name}")

    def _transcribe_wav_file(recording_path: str) -> Optional[str]:

        nonlocal model
        if not os.path.exists(recording_path):
            raise FileNotFoundError(f"{recording_path} not found.")

        with catch_time(function_name='whisper.load_audio'):
            audio = whisper.load_audio(recording_path)
        with catch_time(function_name='whisper.pad_or_trim'):
            audio = whisper.pad_or_trim(audio)
        with catch_time(function_name='whisper.log_mel_spectrogram'):
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

        options = whisper.DecodingOptions(
            language=config.FROM_LANG, fp16=True if device == 'cuda' else False)
        with catch_time(function_name='whisper.decode'):
            result = whisper.decode(model, mel, options)

        if result.no_speech_prob > 0.95:
            print("No speech detected in the recording.")
            return ""
        else:
            print(f"transcribed {recording_path}")
            return result.text
    return _transcribe_wav_file


def extract_sentences(text):
    # This regex captures most typical end-of-sentence punctuations (i.e., .!?).
    # It also tries to account for things like "Mr." or "Dr." so they aren't mistakenly split.
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    sentences = re.split(pattern, text)
    # Removing any empty sentences and stripping leading/trailing spaces
    sentences = [s.strip() for s in sentences if s]
    return sentences


async def put_on_queue(job_data, queue):
    from bullmq import Job
    job = await queue.add('ml-job', job_data)
    job_id = job.id
    job = await Job.fromId(queue, job_id)
    print(f"pushed on {queue.name} queue")
    return await job.getState(), job.returnvalue


def append_to_file(filepath: str, text, sep: str = '\n\n'):
    with open(filepath, 'a') as f_out:
        f_out.write(f"{text}{sep}")


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", 1000)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def pickle_object(to_dump, path: str):
    with open(path, 'wb') as f_out:
        pickle.dump(to_dump, f_out)


def unpickle_object(path: str):
    with open(path, 'rb') as f_in:
        data = pickle.load(f_in)
    return data


def download_and_convert_youtube_link(youtube_url, output_filename, video_name):
    # Download YouTube video
    youtube = YouTube(youtube_url)
    stream = youtube.streams.first()
    stream.download(filename=f'data/examples/{video_name}/video.mp4')

    # Convert video to .wav
    audio = AudioSegment.from_file(f'data/examples/{video_name}/video.mp4')
    audio.export(output_filename, format='wav')


def get_recording_filepath(session_id: str, client_id: str,
                           next_index: int) -> str:
    return f"{paths.RECORDINGS_RESULTS_DIR}/{session_id}_{client_id}_" \
           f"{next_index}.wav"


def get_file_content(filepath: str, last: int = None) -> str:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f_out:
            text = f_out.read()
            if last is not None:
                text = ' '.join(extract_sentences(text)[-last:])
        return text
    else:
        return ''


def clean_files(session_id):
    for folder in [paths.LLM_TRANSLATIONS_RESULTS_DIR,
                   paths.TRANSCRIPTIONS_RESULTS_DIR,
                   paths.TRANSLATIONS_RESULTS_DIR]:
        file_path = f"{folder}/{session_id}.txt"
        open(file_path, 'w').close()
        print(f"cleaned: {file_path}")


def return_default(result, default):
    return result if result is not None else default