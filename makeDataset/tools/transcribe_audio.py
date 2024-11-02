import re
import numpy as np
import os
from dotenv import load_dotenv
import openai
import glob

#get api key from env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

#openai client:
client = openai.Client(api_key=API_KEY)


import re

def transcribe_audio_with_timestamps(audio_path):
    print(f"Transcribing {audio_path}...")
    audio_file = open(audio_path, "rb")
    
    # Transcribe the audio file
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="srt",
    )
    
    #print("Raw Transcript:")
    #print(transcript)

    #save transcript to srt folder:
    with open(f"makeDataset/tools/raw_srt/{os.path.basename(audio_path)[:-4]}.srt", "w") as f:
        f.write(transcript)
    
    return transcript  # Return the list of sentences

def transcribe_all_files(audio_dir: str):
    print(f"Transcribing all audio files in {audio_dir}...")
    audio_files = glob.glob(audio_dir+"/*.wav")
    
    for audio_file in audio_files:
            transcribe_audio_with_timestamps(audio_file)
