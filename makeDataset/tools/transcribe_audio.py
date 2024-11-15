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
    
    # Explicitly check for each expected file
    expected_files = [os.path.join(audio_dir, f"{i}.wav") for i in range(1, 10)]
    
    print("\nVerifying each file individually:")
    valid_files = []
    missing_files = []
    unreadable_files = []
    
    for file_path in expected_files:
        print(f"Checking {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            missing_files.append(file_path)
            continue
            
        if not os.path.isfile(file_path):
            print(f"Not a file: {file_path}")
            missing_files.append(file_path)
            continue
            
        try:
            # Try to open and read the file
            with open(file_path, 'rb') as f:
                f.read(1)
            print(f"Verified readable: {file_path}")
            valid_files.append(file_path)
        except Exception as e:
            print(f"Unreadable: {file_path} - Error: {str(e)}")
            unreadable_files.append((file_path, str(e)))
    
    print("\nVerification Summary:")
    print(f"Expected files: {len(expected_files)}")
    print(f"Valid files: {len(valid_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Unreadable files: {len(unreadable_files)}")
    
    if missing_files:
        print("\nMissing files:")
        for f in missing_files:
            print(f"- {f}")
            
    if unreadable_files:
        print("\nUnreadable files:")
        for f, error in unreadable_files:
            print(f"- {f}: {error}")
    
    # Directory contents check
    print("\nActual directory contents:")
    all_files = os.listdir(audio_dir)
    print(f"Total files in directory: {len(all_files)}")
    print(f"Files: {all_files}")
    
    # Proceed with transcription for valid files
    for audio_file in valid_files:
        print(f"\nStarting transcription of {audio_file}")
        try:
            transcribe_audio_with_timestamps(audio_file)
            print(f"Completed transcription of {audio_file}")
        except Exception as e:
            print(f"Error transcribing {audio_file}: {str(e)}")
