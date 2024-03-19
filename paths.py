import os
from typing import AnyStr


def create_folder(folder_path: AnyStr) -> AnyStr:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


PROJECT_DIR = f'{os.path.dirname(os.path.abspath(__file__))}'
DATA_DIR = create_folder(f'{PROJECT_DIR}/data')
RESULTS_DIR = create_folder(f'{PROJECT_DIR}/results')
LLM_TRANSLATIONS_RESULTS_DIR = create_folder(f'{RESULTS_DIR}/llm_translations')
TRANSLATIONS_RESULTS_DIR = create_folder(f'{RESULTS_DIR}/translations')
TRANSCRIPTIONS_RESULTS_DIR = create_folder(f'{RESULTS_DIR}/transcriptions')
RECORDINGS_RESULTS_DIR = create_folder(f'{RESULTS_DIR}/recordings')
OUTPUT_RECORDINGS_RESULTS_DIR = create_folder(f'{RESULTS_DIR}/output_recordings')
