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
    audio_duration = transcript_dict.get("duration", 0)
    
    try:
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
        
        # If no valid timestamps were found or there was an error, create a single segment
        if not raw_timestamps or None in [start_time, end_time]:
            raise Exception("Could not find reliable word timestamps")
        
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
        
        # Modified section for SRT content generation
        srt_content = []
        segment_counter = 1
        
        for i, ((sentence, ending), (start_time, end_time)) in enumerate(zip(sentences_with_endings, adjusted_timestamps)):
            duration = end_time - start_time

            print(f"Duration: {duration}")
            
            # Check if segment is longer than 8 seconds and contains a comma
            if duration > 8 and ',' in sentence:
                # Add debug print
                print(f"Found long sentence ({duration} seconds): {sentence}")
                
                # Find all comma positions
                comma_positions = [i for i, char in enumerate(sentence) if char == ',']
                print(f"Comma positions: {comma_positions}")
                
                if comma_positions:
                    # Take the middle comma
                    middle_comma_index = comma_positions[len(comma_positions) // 2]
                    first_part = sentence[:middle_comma_index + 1]
                    second_part = sentence[middle_comma_index + 1:].strip()
                    print(f"Split into:\n1: {first_part}\n2: {second_part}")
                    
                    # Find the words before and after the comma
                    words_before = first_part.split()[-1].lower().rstrip(',')
                    words_after = second_part.split()[0].lower()
                    print(f"Looking for words: '{words_before}' and '{words_after}'")
                    
                    # Find timestamps for these words
                    split_time = None
                    end_first = None
                    start_second = None
                    
                    # Create windows of words around the comma
                    before_window = ' '.join(first_part.split()[-3:]).lower()  # Last 3 words before comma
                    after_window = ' '.join(second_part.split()[:3]).lower()   # First 3 words after comma
                    print(f"Looking in windows:\nBefore: {before_window}\nAfter: {after_window}")
                    
                    # Clean windows for comparison by removing punctuation
                    before_window = re.sub(r'[.,!?]', '', before_window).lower()
                    after_window = re.sub(r'[.,!?]', '', after_window).lower()
                    print(f"Looking in cleaned windows:\nBefore: {before_window}\nAfter: {after_window}")
                    
                    # Build a string of all words from transcript for searching
                    transcript_text = ' '.join(
                        word_data["word"].lower() if isinstance(word_data, dict) else word_data.word.lower() 
                        for word_data in transcript_dict["words"]
                    )
                    
                    # Find the best matching positions
                    for i, word_data in enumerate(transcript_dict["words"]):
                        if i >= 2:
                            # Clean transcript window
                            transcript_before = ' '.join(
                                re.sub(r'[.,!?]', '', transcript_dict["words"][j]["word"].lower())
                                for j in range(i-2, i+1)
                            )
                            if end_first is None and before_window in transcript_before:
                                end_first = word_data["end"] if isinstance(word_data, dict) else word_data.end
                                print(f"Found first window match: {transcript_before} at {end_first}")
                        
                        if i <= len(transcript_dict["words"]) - 3:
                            # Clean transcript window
                            transcript_after = ' '.join(
                                re.sub(r'[.,!?]', '', transcript_dict["words"][j]["word"].lower())
                                for j in range(i, i+3)
                            )
                            if start_second is None and after_window in transcript_after:
                                start_second = word_data["start"] if isinstance(word_data, dict) else word_data.start
                                print(f"Found second window match: {transcript_after} at {start_second}")
                        
                        if end_first is not None and start_second is not None:
                            break
                    
                    if end_first is not None and start_second is not None:
                        split_time = (end_first + start_second) / 2
                        print(f"Splitting at time: {split_time}")
                        
                        # Add first part
                        srt_content.extend([
                            str(segment_counter),
                            f"{format_timestamp(start_time)} --> {format_timestamp(split_time)}",
                            first_part,
                            ""
                        ])
                        segment_counter += 1
                        
                        # Add second part
                        srt_content.extend([
                            str(segment_counter),
                            f"{format_timestamp(split_time)} --> {format_timestamp(end_time)}",
                            second_part + ending,
                            ""
                        ])
                        segment_counter += 1
                        continue
                    else:
                        print("Could not find reliable word timestamps for splitting")
            
            # If no split needed or possible, add the whole sentence
            srt_content.extend([
                str(segment_counter),
                f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}",
                sentence,
                ""
            ])
            segment_counter += 1
        
        # Save to SRT file
        srt_filename = f"{os.path.basename(audio_path)[:-4]}.srt"
        srt_path = os.path.join(Config.SRT_DIR, srt_filename)
        os.makedirs(os.path.dirname(srt_path), exist_ok=True)
        
        with open(srt_path, "w", encoding='utf-8') as f:
            f.write("\n".join(srt_content))
        
        return "\n".join(srt_content)
        
    except Exception as e:
        print(f"Warning: Falling back to single segment due to error: {str(e)}")
        # Create single segment SRT
        srt_content = [
            "1",
            f"00:00:00,000 --> {format_timestamp(audio_duration)}",
            original_text,
            ""
        ]
        
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
