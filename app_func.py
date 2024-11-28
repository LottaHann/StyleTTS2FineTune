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
from makeDataset.tools.resampler import resample_audio
from config import Config  # Import the global Config

def _print_segmented_audio_lengths() -> None:
    """Print the lengths of the segmented audio files and their average length using pydub"""
    from pydub import AudioSegment
    lengths = []
    segmented_dir = Config.SEGMENTED_AUDIO_DIR
    for file in os.listdir(segmented_dir):
        audio = AudioSegment.from_file(os.path.join(segmented_dir, file))
        lengths.append(len(audio))
    print(lengths)
    print(f"Number of segmented audio files: {len(lengths)}")
    print(f"Total length of segmented audio files (in minutes): {sum(lengths) / 60000}")
    print(f"Average length (in milliseconds): {sum(lengths) / len(lengths)}")

class AudioProcessor:
    @staticmethod
    def process_dataset(audio_dir: str) -> None:
        """Process audio files and create dataset"""        
        # Create necessary directories first
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.TRAINING_DATA_DIR, exist_ok=True)
        
        print("Processing audio segments...")
        total_segments, total_duration = process_audio_segments()
        print(f"Processed {total_segments} segments with total duration of {total_duration/1000:.2f} seconds")

        print("Resampling audio files...")
        AudioProcessor._resample_audio_files()

        print("Phonemizing transcriptions...")
        AudioProcessor._phonemize_transcriptions()

        _print_segmented_audio_lengths()

    @staticmethod
    def _phonemize_transcriptions() -> None:
        """Handle phonemization of transcriptions"""
        # Ensure directories exist
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.TRAINING_DATA_DIR, exist_ok=True)
        
        input_file = os.path.join(Config.TRAINING_DATA_DIR, "output.txt")
        train_output_file = os.path.join(Config.DATA_DIR, "train_list.txt")
        val_output_file = os.path.join(Config.DATA_DIR, "val_list.txt")
        
        # Verify input file exists before proceeding
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        success = phonemize_transcriptions(
            input_file=input_file,
            train_output_file=train_output_file,
            val_output_file=val_output_file,
            language="en-us"
        )
        
        if not success:
            raise Exception("Phonemization failed")
        
    @staticmethod
    def _resample_audio_files() -> None:
        """Resample audio files to target sample rate (24kHz). Copy to temp and then back in segmented_audio_dir"""
        temp_dir = Config.TEMP_DIR
        segmented_audio_dir = Config.SEGMENTED_AUDIO_DIR
        target_sr = 24000

        try:
            # clear temp dir
            FileHandler.clear_directory(temp_dir)
            
            print(f"Resampling {len(os.listdir(segmented_audio_dir))} audio files to {target_sr} Hz...")
            
            for file in os.listdir(segmented_audio_dir):
                input_path = os.path.join(segmented_audio_dir, file)
                output_path = os.path.join(temp_dir, file)
                
                if not resample_audio(input_path, output_path, target_sr):
                    raise Exception(f"Failed to resample file: {file}")

            # delete segmented_audio_dir contents
            FileHandler.clear_directory(segmented_audio_dir)

            # copy temp contents to segmented_audio_dir
            FileHandler.copy_files(temp_dir, segmented_audio_dir)
            
        except Exception as e:
            print(f"Error during resampling process: {str(e)}")
            raise  # Re-raise the exception to be caught by the caller


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
        moved_files = set()
        skipped_files = []
        
        for root, _, files in os.walk(src_folder):
            for file in files:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)
                
                try:
                    if os.path.exists(dest_path):
                        print(f"Warning: Destination file already exists: {dest_path}")
                        # Generate new unique name
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_path):
                            new_name = f"{base}_{counter}{ext}"
                            dest_path = os.path.join(dest_folder, new_name)
                            counter += 1
                        print(f"Using alternative name: {os.path.basename(dest_path)}")
                    
                    shutil.move(src_path, dest_path)
                    moved_files.add(file)
                except Exception as e:
                    print(f"Error moving file {file}: {str(e)}")
                    skipped_files.append((file, str(e)))
                    continue
        
        if skipped_files:
            print("\nSkipped files during move operation:")
            for file, error in skipped_files:
                print(f"- {file}: {error}")
            print(f"Total files skipped: {len(skipped_files)}")
        
        print(f"Successfully moved {len(moved_files)} files")

    @staticmethod
    def copy_files(src_folder: str, dest_folder: str) -> None:
        """Copy all files from source to destination folder"""
        copied_files = set()
        skipped_files = []
        
        for root, _, files in os.walk(src_folder):
            for file in files:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)
                
                try:
                    if os.path.exists(dest_path):
                        print(f"Warning: Destination file already exists: {dest_path}")
                        # Generate new unique name
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_path):
                            new_name = f"{base}_{counter}{ext}"
                            dest_path = os.path.join(dest_folder, new_name)
                            counter += 1
                        print(f"Using alternative name: {os.path.basename(dest_path)}")
                    
                    shutil.copy2(src_path, dest_path)
                    copied_files.add(file)
                except Exception as e:
                    print(f"Error copying file {file}: {str(e)}")
                    skipped_files.append((file, str(e)))
                    continue
        
        if skipped_files:
            print("\nSkipped files during copy operation:")
            for file, error in skipped_files:
                print(f"- {file}: {error}")
            print(f"Total files skipped: {len(skipped_files)}")
        
        print(f"Successfully copied {len(copied_files)} files")

def clean_exit() -> None:
    """Clean up directories and exit"""
    directories = [
        Config.AUDIO_DIR,
        Config.SRT_DIR,
        Config.SEGMENTED_AUDIO_DIR,
        Config.TRAINING_DATA_DIR,
        Config.TEMP_DIR
    ]
    
    for directory in directories:
        FileHandler.clear_directory(directory)
    
    print("Exiting the application...")
    exit(0)

def save_dataset(voice_id: str) -> Optional[str]:
    """Save and upload the dataset"""
    print("Saving the dataset...")
    try:
        # Verify required files exist before proceeding
        required_files = [
            os.path.join(Config.DATA_DIR, 'train_list.txt'),
            os.path.join(Config.DATA_DIR, 'val_list.txt'),
            os.path.join(Config.DATA_DIR, 'OOD_texts.txt')
        ]
        
        print("\nVerifying required files:")
        for file_path in required_files:
            print(f"Checking {file_path}...")
            if not os.path.exists(file_path):
                print(f"Directory contents of {os.path.dirname(file_path)}:")
                print(os.listdir(os.path.dirname(file_path)))
                raise FileNotFoundError(f"Required file not found: {file_path}")
            else:
                print(f"File exists: {file_path}")

        helsinki_tz = pytz.timezone('Europe/Helsinki')
        timestamp = datetime.now(helsinki_tz).strftime("%d_%m_%Y_%H_%M")
        
        zip_path = os.path.join(Config.TEMP_DIR, f"dataset_{voice_id}_{timestamp}.zip")
        destination_blob_name = f"datasets/dataset_{voice_id}_{timestamp}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add wavs directory
            for root, _, files in os.walk(Config.FINAL_WAVS_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('wavs', file)
                    zip_file.write(file_path, arcname)

            # Add raw_wavs directory
            for root, _, files in os.walk(Config.FINAL_RAW_WAVS_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('raw_wavs', file)
                    zip_file.write(file_path, arcname)

            # Add text files
            zip_file.write(os.path.join(Config.DATA_DIR, 'train_list.txt'), 'train_list.txt')
            zip_file.write(os.path.join(Config.DATA_DIR, 'val_list.txt'), 'val_list.txt')
            zip_file.write(os.path.join(Config.DATA_DIR, 'OOD_texts.txt'), 'OOD_texts.txt')

        # Upload to Firebase
        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(zip_path)
        
        print(f"Dataset saved and uploaded successfully: {destination_blob_name}")
        return destination_blob_name

    except Exception as e:
        print(f"Error saving dataset: {str(e)}")
        return None

def process_uploaded_wavs(audio_dir: str, dataset_percentage: float) -> None:
    """Process uploaded WAV files and prepare them for dataset creation"""
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # Calculate how many files to keep based on dataset_percentage
    print(f"Number of WAV files: {len(wav_files)}")
    num_files_to_keep = max(1, int(len(wav_files) * dataset_percentage))
    print(f"Keeping {num_files_to_keep} files")

    # If we have more files than needed, remove excess files
    if len(wav_files) > num_files_to_keep:
        files_to_remove = wav_files[num_files_to_keep:]
        for file_name in files_to_remove:
            file_path = os.path.join(audio_dir, file_name)
            os.remove(file_path)