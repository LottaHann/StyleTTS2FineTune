import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import shutil
import firebase_admin
from firebase_admin import credentials
import requests
from training_texts import TRAINING_TEXTS
from pydub import AudioSegment
import time

from app_func import (
    AudioProcessor,
    FileHandler,
    ModelHandler,
    ConfigHandler,
    clean_exit,
    save_dataset,
    finetune_process
)
from download_model import download_model

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration"""
    # Directory paths
    AUDIO_DIR = "./makeDataset/tools/audio"
    DEFAULT_CONFIG_PATH = './default_config.yml'
    FINAL_CONFIG_PATH = './model/StyleTTS2/Configs/config_ft.yml'
    RAW_SRT_DIR = './makeDataset/tools/raw_srt'
    SRT_DIR = './makeDataset/tools/srt'
    WAV_DIR_FINETUNING = './model/StyleTTS2/Data/wavs'
    TEMP_DIR = './temp'
    
    # Firebase configuration
    FIREBASE_CREDENTIALS = 'audiobookgen-firebase-adminsdk-mhp3c-3ecc20514f.json'
    FIREBASE_BUCKET = 'audiobookgen.appspot.com'
    
    # ElevenLabs configuration
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    
    @classmethod
    def initialize_firebase(cls) -> None:
        """Initialize Firebase with credentials"""
        if not cls.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY environment variable is not set")
            
        cred = credentials.Certificate(cls.FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred, {
            'storageBucket': cls.FIREBASE_BUCKET
        })

# Initialize Flask and Firebase
app = Flask(__name__)
Config.initialize_firebase()

def generate_training_audio(voice_id: str) -> None:
    """Generate training audio using ElevenLabs API"""
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": Config.ELEVENLABS_API_KEY
    }

    # Ensure audio directory exists and is empty
    FileHandler.clear_directory(Config.AUDIO_DIR)
    os.makedirs(Config.AUDIO_DIR, exist_ok=True)
    print(f"Audio directory contents (should be empty): {os.listdir(Config.AUDIO_DIR)}")

    failed_generations = []
    successful_generations = []

    for idx, text in enumerate(TRAINING_TEXTS, 1):
        output_path = os.path.join(Config.AUDIO_DIR, f"{idx}.wav")
        temp_mp3_path = os.path.join(Config.AUDIO_DIR, f"temp_{idx}.mp3")

        try:
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.1,
                    "use_speaker_boost": False
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

            successful_generations.append(idx)
            print(f"Successfully generated audio for text {idx}")

        except requests.exceptions.RequestException as e:
            failed_generations.append((idx, f"API request failed: {str(e)}"))
            print(f"Failed to generate audio {idx}: API error - {str(e)}")
        except Exception as e:
            failed_generations.append((idx, f"Processing failed: {str(e)}"))
            print(f"Failed to generate audio {idx}: {str(e)}")
        finally:
            # Clean up temp MP3 file if it exists
            if os.path.exists(temp_mp3_path):
                os.remove(temp_mp3_path)

    # Final verification with retries
    max_retries = 3
    for attempt in range(max_retries):
        actual_files = os.listdir(Config.AUDIO_DIR)
        print(f"\nFinal Verification (Attempt {attempt + 1}/{max_retries}):")
        print(f"Expected files: {len(TRAINING_TEXTS)}")
        print(f"Successfully generated: {len(successful_generations)}")
        print(f"Actually present in directory: {len(actual_files)}")
        print(f"Files present: {actual_files}")
        
        if len(actual_files) == len(TRAINING_TEXTS):
            # Verify each file is readable
            all_files_readable = True
            for filename in actual_files:
                file_path = os.path.join(Config.AUDIO_DIR, filename)
                try:
                    with open(file_path, 'rb') as f:
                        # Try to read the first byte
                        f.read(1)
                except Exception as e:
                    print(f"File {filename} is not readable: {str(e)}")
                    all_files_readable = False
            
            if all_files_readable:
                print("All files verified and readable")
                time.sleep(2)  # Wait for file system to stabilize
                return
        
        print(f"Missing files: {set(f'{i}.wav' for i in range(1, len(TRAINING_TEXTS) + 1)) - set(actual_files)}")
        time.sleep(2)  # Wait before retry
    
    raise Exception(
        f"Audio generation verification failed after {max_retries} attempts. "
        f"Expected {len(TRAINING_TEXTS)} files, but {len(actual_files)} were present."
    )

@app.route('/finetune', methods=['POST'])
def finetune():
    """Handle finetuning request"""
    try:
        data = request.json
        voice_id = data.get('voice_id')
        config_url = data.get('config_url')
        dataset_only = data.get('dataset_only', False)

        if not voice_id:
            return jsonify({"error": "Missing voice_id"}), 400

        # Clear directories
        for dir_path in [Config.RAW_SRT_DIR, Config.SRT_DIR]:
            FileHandler.clear_directory(dir_path)

        # Generate training audio using ElevenLabs
        generate_training_audio(voice_id)
        
        # Process the generated audio files
        AudioProcessor.process_dataset(Config.AUDIO_DIR)
        
        # Create dataset
        _create_dataset()

        if dataset_only:
            return _handle_dataset_only()

        # Handle configuration and model
        _handle_configuration(config_url)
        
        # Train model
        download_model()
        ModelHandler.run_finetune(voice_id)
        ModelHandler.save_finetuned_model(voice_id)
        FileHandler.clear_directory(Config.TEMP_DIR)

        return jsonify({"message": f"Model for {voice_id} trained successfully"}), 200

    except Exception as e:
        clean_exit()
        return jsonify({"error": str(e)}), 500

def _create_dataset() -> None:
    """Create and organize dataset structure"""
    # Clear and create directories
    FileHandler.clear_directory('model/StyleTTS2/Data')
    os.makedirs(Config.WAV_DIR_FINETUNING, exist_ok=True)
    FileHandler.clear_directory(Config.WAV_DIR_FINETUNING)
    
    # Move segmented files
    FileHandler.move_files('makeDataset/tools/segmentedAudio', Config.WAV_DIR_FINETUNING)
    FileHandler.move_files('makeDataset/tools/trainingdata', 'model/StyleTTS2/Data')
    shutil.copy('./OOD_texts.txt', './model/StyleTTS2/Data')

def _handle_dataset_only() -> tuple:
    """Handle dataset-only mode"""
    dataset_path = save_dataset()
    if not dataset_path:
        clean_exit()
        return jsonify({"error": "Failed to save or upload dataset"}), 500
    
    FileHandler.clear_directory(Config.TEMP_DIR)
    return jsonify({
        "message": "Dataset created and uploaded successfully",
        "dataset_path": dataset_path
    }), 200

def _handle_configuration(config_url: str) -> None:
    """Handle configuration file setup"""
    if config_url:
        custom_config_path = os.path.join(Config.TEMP_DIR, 'custom_config.yml')
        FileHandler.download_file(config_url, custom_config_path)
        ConfigHandler.update_config(Config.DEFAULT_CONFIG_PATH, custom_config_path)
    else:
        shutil.copy(Config.DEFAULT_CONFIG_PATH, Config.FINAL_CONFIG_PATH)

@app.route('/stop_finetune', methods=['POST'])
def stop_finetune():
    """Stop the finetuning process"""
    if finetune_process is None:
        return jsonify({"error": "No finetuning process is running"}), 400
    
    finetune_process.terminate()
    return jsonify({"message": "Finetuning process stopped"}), 200

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)