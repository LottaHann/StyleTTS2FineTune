import re
import numpy as np
import os
from dotenv import load_dotenv
import openai
import glob
import time
from config import Config

#get api key from env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

#openai client:
client = openai.Client(api_key=API_KEY)


import re

def transcribe_audio_with_timestamps(audio_path, original_text):
    print(f"Transcribing {audio_path}...")
    audio_file = open(audio_path, "rb")
    
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        extra_body={"timestamp_granularities": ["word"]},
        language="en",
        prompt=original_text
    )
    
    transcript_dict = transcript.model_dump() if hasattr(transcript, 'model_dump') else transcript
    sentences_with_endings = []
    current_sentence = ""
    i = 0
    while i < len(original_text):
        if original_text[i:i+3] == "...":
            current_sentence += "..."
            sentences_with_endings.append((current_sentence.strip(), "..."))
            current_sentence = ""
            i += 3
        elif original_text[i] in ".!?":
            current_sentence += original_text[i]
            sentences_with_endings.append((current_sentence.strip(), original_text[i]))
            current_sentence = ""
            i += 1
        else:
            current_sentence += original_text[i]
            i += 1
    
    if current_sentence.strip():
        sentences_with_endings.append((current_sentence.strip(), ""))
    
    sentences = [s[0] for s in sentences_with_endings]
    
    # Create a copy of words that we'll remove from as we process sentences
    remaining_words = transcript_dict["words"].copy()
    
    # First pass: get raw timestamps for each sentence
    raw_timestamps = []
    for sentence in sentences:
        first_word = sentence.split()[0].lower()
        last_word = sentence.split()[-1].lower()
        # Remove any punctuation from last_word
        last_word = re.sub(r'[.,!?]', '', last_word)

        print(f"First word: {first_word}, Last word: {last_word}, Sentence: {sentence}")
        start_time = None
        end_time = None
        
        for word_data in remaining_words[:]:
            word = word_data.word if hasattr(word_data, 'word') else word_data["word"]
            start = word_data.start if hasattr(word_data, 'start') else word_data["start"]
            end = word_data.end if hasattr(word_data, 'end') else word_data["end"]
            
            # Clean the word for comparison
            cleaned_word = re.sub(r'[.,!?]', '', word.lower().strip())
            
            if start_time is None and cleaned_word.startswith(first_word):
                start_time = start
                print(f"Located first word: {word} at {start_time}")
            if cleaned_word == last_word:  # Changed from endswith to exact match
                end_time = end
                print(f"Located last word: {word} at {end_time}")
                if start_time is not None:
                    cutoff_index = remaining_words.index(word_data) + 1
                    remaining_words = remaining_words[cutoff_index:]
                    break
        
        if start_time is not None and end_time is not None:
            raw_timestamps.append((start_time, end_time))
    
    # Second pass: adjust timestamps
    adjusted_timestamps = []
    audio_duration = transcript_dict.get("duration", raw_timestamps[-1][1])
    
    for i in range(len(raw_timestamps)):
        if i == 0:  # First sentence
            start = 0
            if len(raw_timestamps) > 1:
                end = (raw_timestamps[0][1] + raw_timestamps[1][0]) / 2
            else:
                end = audio_duration
        elif i == len(raw_timestamps) - 1:  # Last sentence
            start = (raw_timestamps[i-1][1] + raw_timestamps[i][0]) / 2
            end = audio_duration
        else:  # Middle sentences
            start = (raw_timestamps[i-1][1] + raw_timestamps[i][0]) / 2
            end = (raw_timestamps[i][1] + raw_timestamps[i+1][0]) / 2
        
        adjusted_timestamps.append((start, end))
    
    # Generate SRT content with adjusted timestamps
    srt_content = []
    for i, ((sentence, ending), (start_time, end_time)) in enumerate(zip(sentences_with_endings, adjusted_timestamps), 1):
        start_str = format_timestamp(start_time)
        end_str = format_timestamp(end_time)
        
        srt_content.extend([
            str(i),
            f"{start_str} --> {end_str}",
            sentence,  # Don't add extra punctuation since we preserved the original
            ""
        ])
    
    # Save to SRT file
    srt_filename = f"{os.path.basename(audio_path)[:-4]}.srt"
    srt_path = os.path.join(Config.SRT_DIR, srt_filename)
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)
    
    with open(srt_path, "w", encoding='utf-8') as f:
        f.write("\n".join(srt_content))
    
    return "\n".join(srt_content)

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def transcribe_all_files(audio_dir, training_texts):
    print(f"Transcribing all audio files in {audio_dir}...")
    
    # Get all wav files in the directory and sort them numerically
    audio_files = sorted(
        glob.glob(os.path.join(audio_dir, "*.wav")),
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )
    
    for i, audio_file in enumerate(audio_files):
        try:
            transcribe_audio_with_timestamps(audio_file, training_texts[i])
            print(f"Completed transcription of {audio_file}")
            time.sleep(0.5)  # Add small delay between API calls
        except Exception as e:
            print(f"Error transcribing {audio_file}: {str(e)}")
