# Translates voice live.
### Stack: Whisper model/API, LLM via https://vellum.ai to refine transcripts and translate and keep history for more context translation quality



### requirements
pip install -r requirements.txt
### torch
pip install torch torchvision torchaudio
### install whisper
pip install git+https://github.com/openai/whisper.git
### install whisperx
pip install git+https://github.com/m-bain/whisperx.git


# run in this order
``python async_processor.py``
``python async_recorder.py``