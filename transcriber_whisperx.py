"""
ONLY WITH GPU
"""
import config
import os
import glob
import whisperx

import paths
from utils.helpers import get_secret, catch_time


def get_models(device, compute_type, language_code):
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    return model, model_a, metadata


if __name__ == '__main___':
    batch_size = 16  # reduce if low on GPU mem
    # find most recent files in a directory
    recordings_dir = os.path.join('recordings', '*')
    # model = whisper.load_model(config.WHISPER_MODEL_NAME)
    device = "cuda"
    compute_type = "int8"
    with catch_time(function_name='get_models'):
        model, model_a, metadata = get_models(device, compute_type, language_code=['en'])

    # list to store which wav files have been transcribed
    transcribed = []
    open(paths.TRANSCRIPT_FILE, 'w').close()
    while True:

        # get most recent wav recording in the recordings directory
        files = sorted(glob.iglob(recordings_dir), key=os.path.getctime, reverse=True)
        if len(files) < 1:
            continue

        latest_recording = files[0]
        latest_recording_filename = latest_recording.split('/')[1]

        if os.path.exists(latest_recording) and not latest_recording in transcribed:
            with catch_time(function_name='load_audio'):
                audio = whisperx.load_audio(latest_recording_filename)

            with catch_time(function_name='model.transcribe'):
                result = model.processor(audio, batch_size=batch_size)
                print(result["segments"])  # before alignment
            with catch_time(function_name='whisperx.align'):
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            print(result["segments"])  # after alignment

            with catch_time(function_name='whisperx.DiarizationPipeline'):
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=get_secret("HUGGINGFACE_TOKEN"), device=device)
            with catch_time(function_name='diarize_model'):
                diarize_segments = diarize_model(audio)
            with catch_time(function_name='assign_word_speakers'):
                result = whisperx.assign_word_speakers(diarize_segments, result)

            print(diarize_segments)
            print(result["segments"])  # segments are now assigned speaker IDs