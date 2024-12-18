import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import shutil
import firebase_admin
from firebase_admin import credentials
import requests
from pydub import AudioSegment
import time

from app_func import (
    AudioProcessor,
    FileHandler,
    clean_exit,
    save_dataset,
    process_uploaded_wavs
)
from download_model import download_model
from makeDataset.tools.transcribe_audio import transcribe_all_files
from config import Config

# Load environment variables from .env file
load_dotenv()

# Initialize Flask and Firebase
app = Flask(__name__)
Config.initialize_directories()
Config.initialize_firebase()


def generate_training_audio(voice_id: str, training_texts: list) -> list:
    """
    Generate training audio using ElevenLabs API
    Returns the filtered training texts that correspond to successfully generated audio files
    """
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": Config.ELEVENLABS_API_KEY
    }

    # Ensure audio directory exists and is empty
    FileHandler.clear_directory(Config.AUDIO_DIR)
    os.makedirs(Config.AUDIO_DIR, exist_ok=True)
    
    successful_generations = []  # Will store tuples of (index, text)

    for idx, text in enumerate(training_texts, 1):  # Start from 1 for filename matching
        output_path = os.path.join(Config.AUDIO_DIR, f"{idx}.wav")
        temp_mp3_path = os.path.join(Config.AUDIO_DIR, f"temp_{idx}.mp3")

        try:
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.45,
                    "similarity_boost": 0.75,
                    "style": 0.00,
                    "use_speaker_boost": True
                }
            }

            # Make request to ElevenLabs API
            response = requests.post(
                f"{Config.ELEVENLABS_API_URL}/{voice_id}",
                json=data,
                headers=headers,
                timeout=30  # Add timeout
            )

            response.raise_for_status()  # Raise exception for bad status codes

            # Save the audio file temporarily as MP3
            with open(temp_mp3_path, 'wb') as f:
                f.write(response.content)

            # Convert MP3 to WAV
            print(f"Converting audio {idx} to WAV format...")
            audio = AudioSegment.from_mp3(temp_mp3_path)
            audio.export(output_path, format='wav')

            # Verify the WAV file exists and has size > 0
            if not os.path.exists(output_path):
                raise Exception(f"Generated WAV file is missing")
            if os.path.getsize(output_path) == 0:
                raise Exception(f"Generated WAV file is empty")

            successful_generations.append((idx, text))
            print(f"Successfully generated audio {idx}.wav")

        except Exception as e:
            print(f"Failed to generate audio {idx}.wav: {str(e)}")
            continue  # Skip to next iteration on failure
        finally:
            if os.path.exists(temp_mp3_path):
                os.remove(temp_mp3_path)

    # Return only the texts that were successfully generated
    return [text for _, text in successful_generations]

@app.route('/finetune', methods=['POST'])
def finetune():
    """Handle dataset creation request"""
    try:
        data = request.json
        voice_id = data.get('voice_id')
        dataset_percentage = data.get('dataset_percentage', 0.05)
        wavs_zip_url = data.get('wavs_zip_url')

        if not voice_id:
            return jsonify({"error": "Missing voice_id"}), 400

        if not 0 < dataset_percentage <= 1:
            return jsonify({"error": "dataset_percentage must be between 0 and 1"}), 400

        # Get the training texts
        from training_texts import get_dataset
        from training_dialogs import get_dialog_array
        training_texts_1 = get_dataset(dataset_percentage)
        training_texts_2 = get_dialog_array(dataset_percentage)
        training_texts = training_texts_1 + training_texts_2
        
        if not training_texts:
            return jsonify({"error": "No training texts generated"}), 500

        # Clear directories
        FileHandler.clear_directory(Config.TRAINING_DATA_DIR)
        FileHandler.clear_directory(Config.SRT_DIR)
        FileHandler.clear_directory(Config.AUDIO_DIR)

        # Handle audio source: either from zip or generate new ones
        if wavs_zip_url:
            # Process uploaded WAV files
            temp_zip_path = os.path.join(Config.TEMP_DIR, 'wavs.zip')
            try:
                print(f"Downloading WAV files from: {wavs_zip_url}")
                FileHandler.download_file(wavs_zip_url, temp_zip_path)
                FileHandler.extract_zip(temp_zip_path, Config.AUDIO_DIR)
                process_uploaded_wavs(Config.AUDIO_DIR, dataset_percentage)
                
                # Filter training_texts based on available WAV files
                wav_files = sorted([f for f in os.listdir(Config.AUDIO_DIR) if f.endswith('.wav')])
                available_indices = [int(f.split('.')[0]) for f in wav_files]
                training_texts = [text for idx, text in enumerate(training_texts, 1) if idx in available_indices]
                
                if not wav_files:
                    raise Exception("No WAV files found in downloaded zip")
                print(f"Successfully processed {len(wav_files)} WAV files")
                
            except Exception as e:
                return jsonify({"error": f"Failed to process WAV files: {str(e)}"}), 500
            finally:
                if os.path.exists(temp_zip_path):
                    os.remove(temp_zip_path)
        else:
            # Generate training audio using ElevenLabs and get filtered texts
            training_texts = generate_training_audio(voice_id, training_texts)

        if not training_texts:
            return jsonify({"error": "No audio files were successfully generated or processed"}), 500

        # Transcribe all audio files
        transcribe_all_files(Config.AUDIO_DIR, training_texts)

        # Process the audio files
        AudioProcessor.process_dataset(Config.AUDIO_DIR)

        # Create and save dataset
        _create_dataset(dataset_percentage)
        dataset_path = save_dataset(voice_id)
        
        if not dataset_path:
            return jsonify({"error": "Failed to save or upload dataset"}), 500

        FileHandler.clear_directory(Config.TEMP_DIR)
        return jsonify({
            "message": "Dataset created and uploaded successfully",
            "dataset_path": dataset_path
        }), 200

    except Exception as e:
        clean_exit()
        return jsonify({"error": str(e)}), 500

def _create_dataset(dataset_percentage: float) -> None:
    """Create and organize dataset structure"""
    try:
        # Clear destination directories first
        FileHandler.clear_directory(Config.FINAL_WAVS_DIR)
        FileHandler.clear_directory(Config.FINAL_RAW_WAVS_DIR)
        
        # Create directories
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.FINAL_WAVS_DIR, exist_ok=True)
        
        # Copy required files first before moving anything
        required_files = [
            ('train_list.txt', Config.DATA_DIR),
            ('val_list.txt', Config.DATA_DIR),
            ('OOD_texts.txt', Config.DATA_DIR)
        ]
        
        # Copy training data files first
        for filename, dest_dir in required_files:
            src_path = os.path.join(Config.TRAINING_DATA_DIR, filename)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(dest_dir, filename))
            elif filename == 'OOD_texts.txt':
                # Special case for OOD_texts.txt which is in the root directory
                shutil.copy2('./OOD_texts.txt', os.path.join(dest_dir, filename))

                # take only dataset_percentage of the OOD_texts.txt
                with open(os.path.join(dest_dir, 'OOD_texts.txt'), 'r') as file:
                    lines = file.readlines()
                with open(os.path.join(dest_dir, 'OOD_texts.txt'), 'w') as file:
                    file.writelines(lines[:int(len(lines) * dataset_percentage)])
        
        # Verify files exist before proceeding with audio files
        for filename, dest_dir in required_files:
            dest_path = os.path.join(dest_dir, filename)
            if not os.path.exists(dest_path):
                raise FileNotFoundError(f"Required file not found after copy: {dest_path}")
        
        # Now handle audio files
        print("Moving audio files...")
        FileHandler.move_files(Config.SEGMENTED_AUDIO_DIR, Config.FINAL_WAVS_DIR)
        FileHandler.copy_files(Config.AUDIO_DIR, Config.FINAL_RAW_WAVS_DIR)
        
        print("Dataset creation completed successfully")
        
    except Exception as e:
        print(f"Error in _create_dataset: {str(e)}")
        raise

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)