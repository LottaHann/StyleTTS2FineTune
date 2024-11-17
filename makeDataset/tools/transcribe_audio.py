import re
import numpy as np
import os
from dotenv import load_dotenv
import openai
import glob
import time

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
    
    # Get all wav files in the directory
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    for audio_file in audio_files:
        try:
            transcribe_audio_with_timestamps(audio_file)
            print(f"Completed transcription of {audio_file}")
        except Exception as e:
            print(f"Error transcribing {audio_file}: {str(e)}")
