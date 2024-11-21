import os
import requests
import shutil
import zipfile
import argparse
from datetime import datetime, timezone
import torch
import yaml
import subprocess
import tarfile
import re
from typing import Optional, List
from firebase_admin import storage
from google.cloud import firestore
import librosa
import soundfile as sf
import pytz

from makeDataset.tools.format_srt import format_srt_file
from makeDataset.tools.transcribe_audio import transcribe_all_files
from makeDataset.tools.phonemized_func import phonemize_transcriptions
from makeDataset.tools.srtsegmenter_func import process_audio_segments

# Global Variables
finetune_process = None

class Config:
    """Configuration constants"""
    STYLETTS2_DIR = "model/StyleTTS2"
    MODEL_DIR = f"{STYLETTS2_DIR}/Models/LJSpeech"
    FINETUNED_DIR = "./finetuned_models"
    FINAL_CONFIG_PATH = f'{STYLETTS2_DIR}/Configs/config_ft.yml'
    DATASET_PATHS = {
        'audio': 'makeDataset/tools/audio',
        'raw_srt': 'makeDataset/tools/raw_srt',
        'srt': 'makeDataset/tools/srt',
        'segmented_audio': 'makeDataset/tools/segmentedAudio',
        'training_data': 'makeDataset/tools/trainingdata',
        'wavs': f'{STYLETTS2_DIR}/Data/wavs',
        'data': f'{STYLETTS2_DIR}/Data'
    }

class AudioProcessor:
    @staticmethod
    def rename_audio_files(audio_dir: str) -> None:
        """Rename audio files sequentially"""
        for i, filename in enumerate(os.listdir(audio_dir)):
            os.rename(
                os.path.join(audio_dir, filename),
                os.path.join(audio_dir, f"{i+1}.wav")
            )

    @staticmethod
    def process_dataset(audio_dir: str) -> None:
        """Process audio files and create dataset"""        
        print("Transcribing audio files...")
        transcribe_all_files(audio_dir)
        
        srt_file_path = "./makeDataset/tools/raw_srt/*.srt"
        format_srt_file(srt_file_path)
        
        print("Segmenting audio files...")
        process_audio_segments()

        AudioProcessor._print_segmented_audio_lengths()
        
        AudioProcessor._phonemize_transcriptions()

    @staticmethod
    def _print_segmented_audio_lengths() -> None:
        """Print the lengths of the segmented audio files and their average length using pydub"""
        from pydub import AudioSegment
        lengths = []
        for file in os.listdir(Config.DATASET_PATHS['segmented_audio']):
            audio = AudioSegment.from_file(os.path.join(Config.DATASET_PATHS['segmented_audio'], file))
            lengths.append(len(audio))
        print(lengths)
        print(f"Number of segmented audio files: {len(lengths)}")
        print(f"Total length of segmented audio files (in minutes): {sum(lengths) / 60000}")
        print(f"Average length (in milliseconds): {sum(lengths) / len(lengths)}")

    @staticmethod
    def _phonemize_transcriptions() -> None:
        """Handle phonemization of transcriptions"""
        parser = argparse.ArgumentParser(description="Phonemize transcriptions.")
        parser.add_argument("--language", type=str, default="en-us")
        parser.add_argument("--input_file", type=str, 
                          default="./makeDataset/tools/trainingdata/output.txt")
        parser.add_argument("--train_output_file", type=str,
                          default="./makeDataset/tools/trainingdata/train_list.txt")
        parser.add_argument("--val_output_file", type=str,
                          default="./makeDataset/tools/trainingdata/val_list.txt")

        args = parser.parse_args()
        phonemize_transcriptions(
            args.input_file, 
            args.train_output_file, 
            args.val_output_file, 
            args.language
        )

class FileHandler:
    @staticmethod
    def download_file(url: str, destination_path: str) -> None:
        """Download file from URL to specified path"""
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file from {url}")
            
        with open(destination_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

    @staticmethod
    def extract_zip(zip_path: str, extract_to: str) -> None:
        """Extract ZIP file to specified directory"""
        temp_extract_dir = f"{extract_to}_temp"
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
            
            for root, _, files in os.walk(temp_extract_dir):
                for file in files:
                    if file.endswith('.wav'):
                        full_file_path = os.path.join(root, file)
                        shutil.move(full_file_path, extract_to)
            
            shutil.rmtree(temp_extract_dir)

    @staticmethod
    def clear_directory(directory: str) -> None:
        """Remove all files in the specified directory"""
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def move_files(src_folder: str, dest_folder: str) -> None:
        """Move all files from source to destination folder"""
        for root, _, files in os.walk(src_folder):
            for file in files:
                full_file_path = os.path.join(root, file)
                shutil.move(full_file_path, dest_folder)

    @staticmethod
    def copy_files(src_folder: str, dest_folder: str) -> None:
        """Copy all files from source to destination folder"""
        for root, _, files in os.walk(src_folder):
            for file in files:
                full_file_path = os.path.join(root, file)
                shutil.copy(full_file_path, dest_folder)

class ModelHandler:
    @staticmethod
    def find_newest_model(directory: str) -> Optional[str]:
        """Find the most recent model file"""
        pattern = re.compile(r'epoch_2nd_(\d{5}).pth')
        model_files = [
            (int(match.group(1)), file)
            for file in os.listdir(directory)
            if (match := pattern.match(file))
        ]
        
        if not model_files:
            return None
            
        newest_model = max(model_files, key=lambda x: x[0])[1]
        return os.path.join(directory, newest_model)

    @staticmethod
    def run_finetune(voice_id: str) -> None:
        """Run the model finetuning process"""
        global finetune_process
        try:
            print("GPU is available:", torch.cuda.is_available())
            
            command = [
                "accelerate", "launch",
                "--mixed_precision=fp16",
                "--num_processes=1",
                "train_finetune_accelerate.py",
                "--config_path", "./Configs/config_ft.yml"
            ]
            
            finetune_process = subprocess.Popen(
                command, 
                cwd=Config.STYLETTS2_DIR
            )
            print("Model finetuning started successfully")
            
            finetune_process.wait()
            print("Calling save_finetuned_model")
            ModelHandler.save_finetuned_model(voice_id)

        except Exception as e:
            print(f"Error during model finetuning: {e}")

    @staticmethod
    def save_finetuned_model(voice_id: str) -> Optional[str]:
        """Save and upload the finetuned model"""
        print("Saving the finetuned model...")
        try:
            newest_model_path = ModelHandler.find_newest_model(Config.MODEL_DIR)
            if not newest_model_path:
                print("No model files found in the directory.")
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination_blob_name = f"finetuned_models/{voice_id}_{timestamp}.tar"
            tar_path = os.path.join(Config.FINETUNED_DIR, f"{voice_id}_{timestamp}.tar")

            os.makedirs(Config.FINETUNED_DIR, exist_ok=True)
            
            # Create tar file
            with tarfile.open(tar_path, "w") as tar:
                tar.add(newest_model_path, arcname=f"{voice_id}.pth")

            # Upload to Firebase
            bucket = storage.bucket()
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(tar_path)
            
            print(f"Model uploaded successfully: {destination_blob_name}")
            return destination_blob_name

        except Exception as e:
            print(f"Error in save_finetuned_model: {e}")
            return None

class ConfigHandler:
    @staticmethod
    def update_config(config_path: str, new_config_path: str) -> None:
        """Update configuration file with new settings"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            with open(new_config_path, 'r') as f:
                new_config = yaml.safe_load(f)

            if not isinstance(config, dict) or not isinstance(new_config, dict):
                raise ValueError("Both config and new_config must be dictionaries.")

            config.update(new_config)

            if os.path.exists(Config.FINAL_CONFIG_PATH):
                os.remove(Config.FINAL_CONFIG_PATH)
            
            with open(Config.FINAL_CONFIG_PATH, 'w') as f:
                yaml.dump(config, f)

        except Exception as e:
            print(f"Error updating config: {e}")
            raise

def clean_exit() -> None:
    """Clean up directories and exit"""
    for directory in Config.DATASET_PATHS.values():
        FileHandler.clear_directory(directory)
    
    print("Exiting the application...")
    exit(0)

def save_dataset(voice_id: str) -> Optional[str]:
    """Save and upload the dataset with 24kHz audio files"""
    print("Saving the dataset...")
    try:
        # Change to Helsinki timezone
        helsinki_tz = pytz.timezone('Europe/Helsinki')
        timestamp = datetime.now(helsinki_tz).strftime("%d_%m_%Y_%H_%M")
        os.makedirs(Config.FINETUNED_DIR, exist_ok=True)
        
        zip_path = os.path.join(Config.FINETUNED_DIR, f"dataset_{voice_id}_{timestamp}.zip")
        destination_blob_name = f"finetuned_models/dataset_{voice_id}_{timestamp}.zip"
        temp_audio_dir = os.path.join(Config.FINETUNED_DIR, "temp_audio")
        os.makedirs(temp_audio_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Process and add wav files
            wav_dir = os.path.join(Config.DATASET_PATHS['data'], 'wavs')
            for root, _, files in os.walk(wav_dir):
                for file in files:
                    # Load and resample audio
                    file_path = os.path.join(root, file)
                    audio, _ = librosa.load(file_path, sr=24000)
                    
                    # Save resampled audio to temp directory
                    temp_path = os.path.join(temp_audio_dir, file)
                    sf.write(temp_path, audio, 24000)
                    
                    # Add to zip
                    arcname = os.path.join('wavs', file)
                    zip_file.write(temp_path, arcname)

            # Add training and validation files with renamed paths
            train_list_path = os.path.join(Config.DATASET_PATHS['data'], 'train_list.txt')
            val_list_path = os.path.join(Config.DATASET_PATHS['data'], 'val_list.txt')
            ood_path = os.path.join(Config.DATASET_PATHS['data'], 'OOD_texts.txt')

            zip_file.write(train_list_path, 'train_data.txt')
            zip_file.write(val_list_path, 'validation_data.txt')
            zip_file.write(ood_path, 'OOD_data.txt')

            # Add raw_wavs
            raw_wavs_dir = os.path.join(Config.DATASET_PATHS['data'], 'raw_wavs')
            for root, _, files in os.walk(raw_wavs_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.join('raw_wavs', file))

        # Clean up temporary directory
        shutil.rmtree(temp_audio_dir)

        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(zip_path)
        
        print(f"Dataset uploaded successfully: {destination_blob_name}")
        return destination_blob_name

    except Exception as e:
        print(f"Error saving dataset: {e}")
        if os.path.exists(temp_audio_dir):
            shutil.rmtree(temp_audio_dir)
        return None