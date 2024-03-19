import asyncio
import paths

from utils.helpers import transcribe_wav_file
from utils.helpers import download_and_convert_youtube_link
from async_process_one_file import process_one_file
import config
import os


async def main():

    model_name = config.WHISPER_MODEL_NAME
    # video_name = "Alex_Ratner"
    # youtube_url = "https://www.youtube.com/watch?v=gcjOCX61_2o&list=PLBgv57GHyNgjl2j8iAkC3-ETLhJW7xjhB"
    # video_name = "deeplearning_ai"
    # youtube_url = "https://www.youtube.com/watch?v=7EcznH0-of8"

    # video_name = "corporate_zoom"
    # youtube_url = "https://www.youtube.com/watch?v=53yPfrqbpkE"

    # video_name = "lecture_mipt"
    # youtube_url = "https://www.youtube.com/watch?v=XOQKKMYEImw&list=PLk4h7dmY2eYFmowaPqjFDzSokiiLq5TkT&index=8"

    # video_name = 'shad'
    # youtube_url = "https://www.youtube.com/watch?v=tuG5bzAPBJI"

    # video_name = 'biotech_medicine'
    # youtube_url = "https://www.youtube.com/watch?v=SbsT52WL7gs"

    video_name = "fintech"
    youtube_url = "https://www.youtube.com/watch?v=Jor-X1vs2Ho"
    project_context = """
Terms:
- Линеар/Linear
- Плейграунд/Playground
- Эйртейбл/Airtable
- ЭНЭФТИ/NFT


Names:
- Вен/Wen
- Егор/Egor
- Уэнчи/WenQi
- Уан/Uan
- Энтони/Anthony
    """

    if not os.path.exists(f"data/examples/{video_name}"):
        os.makedirs(f"data/examples/{video_name}")
    wav_file = f"data/examples/{video_name}/audio.wav"

    download_and_convert_youtube_link(youtube_url, wav_file, video_name)
    # transcript = transcribe_wav_file(wav_file, model_name)
    await process_one_file(wav_file, session_id=video_name, project_context=project_context)

    # with open(f"data/examples/{video_name}/transcript.txt", "w") as file:
    #     file.write(transcript)``

    # translated_text = translator.translate(transcript)
    # with open(f"data/examples/{video_name}/translation.txt", "w") as file:
    #     file.write(translated_text)

if __name__ == "__main__":
    asyncio.run(main())