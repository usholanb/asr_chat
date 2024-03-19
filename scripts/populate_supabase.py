import os
from pytube import YouTube
from pydub import AudioSegment
from pydub.silence import split_on_silence
from supabase import create_client 
import config
import whisper
import pyaudio
import wave

supabase_url = "https://xvhcqomaculhentrldfp.supabase.co"
# supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFpbWR0aWhnaGJoYWVtYmx5bW5mIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODYxMTQ0NzEsImV4cCI6MjAwMTY5MDQ3MX0.MvnlWFEwhU_2EDw7J_Yb1pJUeB0e5cY5aRoOWVXdMyE"
# supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh2aGNxb21hY3VsaGVudHJsZGZwIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTE0NDk2NDIsImV4cCI6MjAwNzAyNTY0Mn0.EUD_adLZeFKo1L0OGXgJNOfwvDxVPe6YspRFyKIP0T0"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh2aGNxb21hY3VsaGVudHJsZGZwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY5MTQ0OTY0MiwiZXhwIjoyMDA3MDI1NjQyfQ.DdEjLjV8gy0Ad-grEPEWi_Nc6wF6pWBwaejOON6nJf4"
supabase_db = create_client(supabase_url, supabase_key)
bucket_name = 'transladar_wav'
all_bucket_file_names  = [f['name'] for f in supabase_db.storage.from_(bucket_name).list() if 'wav' in f['name']]
# supabase = create_supabase(url = supabase_url, key = supabase_key)
model = whisper.load_model('tiny')


def create_source(youtube_url, video_name, original_lang):
    # Download YouTube video as mp4
    yt = YouTube(youtube_url)
    video = yt.streams.first()
    video.download(output_path='/tmp', filename=f'{video_name}.mp4')

    # Convert mp4 to wav
    audio = AudioSegment.from_file(f'/tmp/{video_name}.mp4')
    audio.export(f'/tmp/{video_name}.wav', format='wav')
    wav_fname = f'{video_name}.wav'
    if wav_fname not in all_bucket_file_names:
        with open(f'/tmp/{video_name}.wav', 'rb') as f:
            supabase_db.storage.from_(bucket_name).upload(
                wav_fname,
                f
            )

    # Create entry in audio_sources table
    # new_id = supabase_db.table('audio_sources').select('id').order('id', ascending=False).execute()['data'][0]['id'] + 1
    supabase_db.table('audio_sources').insert({
        'context': '',
        'url': youtube_url,
        'wav_link': f'{video_name}.wav',
        'original_lang': original_lang,
        'name': video_name,
    }).execute()


def source2chunks(video_name, orig_lang):
    # 1) Find a source in the audio_sources table with that name
    source = supabase_db.table('audio_sources').select('*').eq('name', video_name).execute().data[0]

    # 2) Download the wav file locally using "wav_link" and the supabase bucket
    wav_link = source['wav_link']
    with open('/tmp/{video_name}.wav', 'wb') as f:
        res = supabase_db.storage.from_(bucket_name).download(wav_link)
        f.write(res)

    # 3) Split the wav into chunks of size ~30 seconds (enforce 20-40 seconds) using pydub split_on_silence
    audio = AudioSegment.from_wav(f'/tmp/{video_name}.wav')
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40, keep_silence=500)
    # Merge chunks under 20 seconds
    merged_chunks = []
    temp_chunk = chunks[0]
    for chunk in chunks[1:]:
        if len(temp_chunk) < 20000:  # If the chunk is less than 20 seconds
            temp_chunk += chunk  # Merge the chunk
        else:
            merged_chunks.append(temp_chunk)
            temp_chunk = chunk
    merged_chunks.append(temp_chunk)  # Append the last chunk
    
    valid_chunks = []
    for chunk in merged_chunks:
        while len(chunk) > 40000: # Split too long chunks into 30s parts
            part = chunk[:30000]
            valid_chunks.append(part)
            chunk = chunk[30000:]
        valid_chunks.append(chunk)
    print(f'Got {len(valid_chunks)} chunks')
    # 4) Insert each chunk into the "audio_chunks" table in supabase
    for i, chunk in enumerate(valid_chunks):
        chunk.export(f'/tmp/{video_name}_chunk{i}.wav', format='wav')
        wav_fname = f'{video_name}_chunk{i}.wav'
        if wav_fname not in all_bucket_file_names:
            with open(f'/tmp/{video_name}_chunk{i}.wav', 'rb') as f:
                supabase_db.storage.from_(bucket_name).upload(
                    wav_fname,
                    f
                )
        audio = whisper.load_audio(f'/tmp/{video_name}_chunk{i}.wav')
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(
            language=orig_lang, fp16=False)
        result = whisper.decode(model, mel, options)

        transcript_tiny = result.text

        supabase_db.table('audio_chunks').insert({
            'source_id': source['id'],
            'wav_link': f'{video_name}_chunk{i}.wav',
            'sequence_number': i,
            'transcript_tiny': transcript_tiny,
        }).execute()

def label_chunks(video_name):
    # Get all chunks for the video
    source = supabase_db.table('audio_sources').select('id').eq('name', video_name).execute().data[0]
    source_id = source['id']
    chunks = supabase_db.table('audio_chunks').select('*').eq('source_id', source_id).execute().data
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    for chunk in chunks:
        # If the chunk already has text in "ideal_en" field, ignore it
        if chunk['ideal_en']:
            continue

        # Download the wav file
        wav_link = chunk['wav_link']
        with open(f'~/chunk.wav', 'wb') as f:
            res = supabase_db.storage.from_(bucket_name).download(wav_link)
            f.write(res)
        
        print(f"{chunk['transcript_tiny']}")
        ideal_en = input(f"Open ~/chunk.wav. Enter the ideal English translation:")

        # Add the "ideal_en" to the table entry
        supabase_db.table('audio_chunks').update({
            'ideal_en': ideal_en,
        }).eq('id', chunk['id']).execute()

    # Terminate PyAudio
    p.terminate()

if __name__ == "__main__":
    orig_lang =  'ru'

    # video_names  = ['shad','biotech_medicine','fintech','web3ru']
    # youtube_link_array = ['https://www.youtube.com/watch?v=tuG5bzAPBJI', "https://www.youtube.com/watch?v=SbsT52WL7gs", 
    #     "https://www.youtube.com/watch?v=Jor-X1vs2Ho", "https://www.youtube.com/watch?v=v7YQJ9ak4x4"]

    video_names = [
        # 'guriev_vvp',
        # 'implicit_feedback',
        # 'multi_criterio_optimization',
        # 'ansible',
        # 'toxicity_detector',
        ]
    youtube_link_array = [
        # 'https://www.youtube.com/watch?v=8IQ1VjTWqlo',
        # 'https://www.youtube.com/watch?v=RN88Kdzsi5Y',
        # 'https://www.youtube.com/watch?v=VpXcGWy1dhc',
        # 'https://www.youtube.com/watch?v=T4VVdkO9sCo',
        # 'https://www.youtube.com/watch?v=m1_zQLjwwWE'
        ]
    # video_name = 'shad'
    # youtube_url = 'https://www.youtube.com/watch?v=tuG5bzAPBJI'

    # video_name = 'biotech_medicine'
    # youtube_url = "https://www.youtube.com/watch?v=SbsT52WL7gs"

    # video_name = "fintech"
    # youtube_url = "https://www.youtube.com/watch?v=Jor-X1vs2Ho"

    # video_name = "web3ru"
    # youtube_url = "https://www.youtube.com/watch?v=v7YQJ9ak4x4"

    # for (video_name, youtube_url) in zip(video_names, youtube_link_array):
    #     create_source(youtube_url, video_name, orig_lang)
    #     source2chunks(video_name, orig_lang)
    
    label_chunks('biotech_medicine')

    