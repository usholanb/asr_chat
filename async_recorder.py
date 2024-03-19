import sounddevice as sd
import wavio as wv

import config
import utils.constants as const
import asyncio
from bullmq import Queue
import paths
from utils.helpers import catch_time, put_on_queue, get_recording_filepath, \
    clean_files


async def main():
    queue = Queue(const.RECORDINGS_QUEUE)
    session_id, client_id = "recording", '1'
    project_context = """
    Terms:


- Уиспер/Whisper
- Гитхаб/Github

Names:
- Уан/Uan
- Егор/Egor
"""
    clean_files(session_id)
    print("start recording")
    index = 0
    while True:
        # Start recorder with the given values of duration and sample frequency
        # PTL Note: I had to change the channels value in the original code to fix a bug
        recording = sd.rec(int(config.DURATION * const.FREQ), samplerate=const.FREQ, channels=1)

        # Record audio for the given number of seconds
        sd.wait()

        recording_filepath = get_recording_filepath(session_id, client_id, index)

        with catch_time(function_name='wv.write'):
            # Convert the NumPy array to audio file
            wv.write(recording_filepath, recording, const.FREQ, sampwidth=2)
            print(f"saved recording to {recording_filepath}")

        job_data = {
            "session_id": session_id,
            "path": recording_filepath,
            "client_id": client_id,
            "project_context": project_context,
            "next_index": index,
        }
        state, return_value = await put_on_queue(job_data=job_data, queue=queue)
        print(f"state, return_value: {state, return_value}")
        index += 1


if __name__ == "__main__":
    asyncio.run(main())