import sounddevice as sd
import wavio as wv
import whisper

import config
import utils.constants as const
import asyncio
from bullmq import Queue
import paths
from utils.helpers import catch_time, put_on_queue, get_recording_filepath,\
    LimitedSizeDict, clean_files, return_default

_session_to_index = LimitedSizeDict(size_limit=const.MAX_SESSIONS_AT_THE_SAME_TIME)


async def process_one_file(wav_filepath, interval_seconds=None,
                           session_id='123', project_context=""):
    interval_seconds = return_default(interval_seconds, config.DURATION)
    queue = Queue(const.RECORDINGS_QUEUE)
    session_id, client_id = session_id, '1'
    audio = whisper.load_audio(wav_filepath, sr=const.FREQ)
    print("start recording")
    clean_files(session_id)

    for chunk_index, i in enumerate(range(0, len(audio), interval_seconds * const.FREQ)):

        recording = audio[i: i + const.FREQ * interval_seconds]

        # Record audio for the given number of seconds
        sd.wait()
        recording_filepath = get_recording_filepath(session_id, client_id, chunk_index)

        with catch_time(function_name='wv.write'):
            # Convert the NumPy array to audio file
            wv.write(recording_filepath, recording, const.FREQ, sampwidth=2)

        job_data = {
            "session_id": session_id,
            "path": recording_filepath,
            "client_id": client_id,
            "project_context": project_context,
            "next_index": chunk_index,
        }
        state, return_value = await put_on_queue(job_data=job_data, queue=queue)
        print(f"state, return_value: {state, return_value}")


if __name__ == "__main__":
    asyncio.run(process_one_file(
        "/Users/uan/PycharmProjects/asr_chat/data/audio.wav",
        session_id="from_file", project_context=""))