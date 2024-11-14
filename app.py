import os
from flask import Flask, request, jsonify
import shutil
import firebase_admin
from firebase_admin import credentials

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
    
    @classmethod
    def initialize_firebase(cls) -> None:
        """Initialize Firebase with credentials"""
        cred = credentials.Certificate(cls.FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred, {
            'storageBucket': cls.FIREBASE_BUCKET
        })

# Initialize Flask and Firebase
app = Flask(__name__)
Config.initialize_firebase()

@app.route('/finetune', methods=['POST'])
def finetune():
    """Handle finetuning request"""
    try:
        data = request.json
        voice_id = data.get('voice_id')
        audio_zip_url = data.get('audio_zip_url')
        config_url = data.get('config_url')
        dataset_only = data.get('dataset_only', False)

        if not voice_id or not audio_zip_url:
            return jsonify({"error": "Missing voice_id or audio_zip_url"}), 400

        # Clear directories
        for dir_path in [Config.RAW_SRT_DIR, Config.SRT_DIR]:
            FileHandler.clear_directory(dir_path)

        # Process audio files
        audio_zip_path = os.path.join(Config.TEMP_DIR, f"{voice_id}_audio.zip")
        _process_audio_files(audio_zip_url, audio_zip_path)
        
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

def _process_audio_files(audio_zip_url: str, audio_zip_path: str) -> None:
    """Process audio files for training"""
    FileHandler.download_file(audio_zip_url, audio_zip_path)
    FileHandler.clear_directory(Config.AUDIO_DIR)
    FileHandler.extract_zip(audio_zip_path, Config.AUDIO_DIR)
    AudioProcessor.process_dataset(Config.AUDIO_DIR)

def _create_dataset() -> None:
    """Create and organize dataset structure"""
    FileHandler.clear_directory('model/StyleTTS2/Data')
    os.makedirs(Config.WAV_DIR_FINETUNING, exist_ok=True)
    FileHandler.clear_directory(Config.WAV_DIR_FINETUNING)

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