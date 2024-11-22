import replicate
import json
from typing import List, Dict
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def get_forced_alignment(audio_path: str, text: str) -> List[Dict]:
    """Get forced alignment data for an audio file and its text"""
    print(f"Getting forced alignment for {audio_path}...")
    
    output = replicate.run(
        "quinten-kamphuis/forced-alignment:566a5a9530375ba0428344b66027520e83f832527bc04c5c4770cea1d3e6fcc7",
        input={
            "audio": open(audio_path, "rb"),
            "script": text
        }
    )
    
    return output

def convert_to_sentences(alignment_data: List[Dict], text: str) -> List[Dict]:
    """Convert word-level alignment to sentence-level alignment"""
    # Split text into sentences
    sentences = re.split('([.!?]+)', text)
    sentences = [''.join(i) for i in zip(sentences[::2], sentences[1::2])]
    
    current_sentence = []
    sentence_alignments = []
    current_text = ""
    
    for word_data in alignment_data:
        current_sentence.append(word_data)
        current_text += word_data["word"] + " "
        
        # Check if we've completed a sentence
        for sentence in sentences:
            if current_text.strip() in sentence:
                if any(punct in word_data["word"] for punct in ['.', '!', '?']):
                    sentence_alignments.append({
                        'text': current_text.strip(),
                        'start': current_sentence[0]['start'],
                        'end': current_sentence[-1]['end']
                    })
                    current_sentence = []
                    current_text = ""
                break
    
    # Add any remaining text as a sentence
    if current_sentence:
        sentence_alignments.append({
            'text': current_text.strip(),
            'start': current_sentence[0]['start'],
            'end': current_sentence[-1]['end']
        })
    
    return sentence_alignments

def create_srt_content(sentences: List[Dict]) -> str:
    """Create SRT format content from sentence alignments"""
    srt_content = ""
    for i, sentence in enumerate(sentences, 1):
        # Convert timestamps to SRT format (HH:MM:SS,mmm)
        start_time = f"{int(sentence['start'] // 3600):02d}:{int((sentence['start'] % 3600) // 60):02d}:{int(sentence['start'] % 60):02d},{int((sentence['start'] % 1) * 1000):03d}"
        end_time = f"{int(sentence['end'] // 3600):02d}:{int((sentence['end'] % 3600) // 60):02d}:{int(sentence['end'] % 60):02d},{int((sentence['end'] % 1) * 1000):03d}"
        
        srt_content += f"{i}\n{start_time} --> {end_time}\n{sentence['text']}\n\n"
    
    return srt_content

def process_audio_files(audio_dir: str, training_texts: List[str]) -> None:
    """Process all audio files and create SRT files using forced alignment"""
    print(f"Processing audio files in {audio_dir}...")
    
    # Create SRT directory if it doesn't exist
    srt_dir = "./makeDataset/tools/srt"
    os.makedirs(srt_dir, exist_ok=True)
    
    # Get sorted list of audio files
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')],
                        key=lambda x: int(x.split('.')[0]))
    
    # Process each audio file with its corresponding text, if available
    for i, audio_file in enumerate(audio_files):
        if i >= len(training_texts):  # Skip if we don't have corresponding text
            break
            
        audio_path = os.path.join(audio_dir, audio_file)
        try:
            # Get forced alignment data
            alignment_data = get_forced_alignment(audio_path, training_texts[i])
            
            # Convert to sentence-level alignment
            sentences = convert_to_sentences(alignment_data, training_texts[i])
            
            # Create SRT content
            srt_content = create_srt_content(sentences)
            
            # Save SRT file
            srt_path = os.path.join(srt_dir, f"{i+1}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
                
            print(f"Created SRT file for {audio_file}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")