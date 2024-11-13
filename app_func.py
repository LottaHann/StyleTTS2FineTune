import os
import requests
import shutil
import zipfile
from makeDataset.tools.format_srt import format_srt_file
from makeDataset.tools.transcribe_audio import transcribe_all_files
from makeDataset.tools.phonemized_func import phonemize_transcriptions
from makeDataset.tools.srtsegmenter_func import process_audio_segments
import argparse
import shutil
from datetime import datetime
import torch
import yaml
import subprocess
import tarfile
from firebase_admin import storage
from google.cloud import firestore
import re
import datetime


finetune_process = None



def makedataset(audio_dir):
    # name audio files
    for i, filename in enumerate(os.listdir(audio_dir)):
        os.rename(os.path.join(audio_dir, filename), os.path.join(audio_dir, f"{i+1}.wav"))
    
    print("Transcribing audio files...")
    transcribe_all_files(audio_dir)

    print("raw srt files have been created")

    srt_file_path = "./makeDataset/tools/raw_srt/*.srt"

    format_srt_file(srt_file_path)

    print("srt files are being formatted")

    print("Segmenting audio files...")
    process_audio_segments()


    # Argument parser
    parser = argparse.ArgumentParser(description="Phonemize transcriptions.")
    parser.add_argument(
        "--language",
        type=str,
        default="en-us",
        help="The language to use for phonemization.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./makeDataset/tools/trainingdata/output.txt",
        help="Path to input transcriptions file.",
    )
    parser.add_argument(
        "--train_output_file",
        type=str,
        default="./makeDataset/tools/trainingdata/train_list.txt",
        help="Path for train_list.txt in the training data folder.",
    )
    parser.add_argument(
        "--val_output_file",
        type=str,
        default="./makeDataset/tools/trainingdata/val_list.txt",
        help="Path for val_list.txt in the training data folder.",
    )

    args = parser.parse_args()

    # Call the phonemization function
    phonemize_transcriptions(args.input_file, args.train_output_file, args.val_output_file, args.language)

    #copy the contents of makeDataset/tools/trainingdata to trainingdata +  datetime
    #shutil.copytree('./makeDataset/tools/trainingdata', f'./trainingdata{datetime.now().strftime("%Y%m%d%H%M%S")}')

    #copy segmented audio files from makeDataset/tools/segmentedAudio to segmentedAudio
    #shutil.copytree('./makeDataset/tools/segmentedAudio', f'./segmentedAudio{datetime.now().strftime("%Y%m%d%H%M%S")}')

def download_file(url, destination_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    else:
        raise Exception(f"Failed to download file from {url}")

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_extract_dir = f"{extract_to}_temp"  # Temporary directory for extraction
        zip_ref.extractall(temp_extract_dir)  # Extract all files to a temp directory

        # Move files from temp directory to target directory
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                if file.endswith('.wav'):  # Only process audio files
                    full_file_path = os.path.join(root, file)
                    shutil.move(full_file_path, extract_to)  # Move to final destination

        # Clean up the temporary directory
        shutil.rmtree(temp_extract_dir)

def clear_directory(directory):
    """Remove all files in the specified directory."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove all files and subdirectories
    os.makedirs(directory, exist_ok=True)  # Recreate the empty directory

def copy_files(src_folder, dest_folder):
    """Copy all files from src_folder to dest_folder."""
    # Clear the destination folder
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)
    
    # Copy all files from src_folder to dest_folder
    for root, _, files in os.walk(src_folder):
        for file in files:
            full_file_path = os.path.join(root, file)
            shutil.copy(full_file_path, dest_folder)

def move_files(src_folder, dest_folder):
    """Move all files from src_folder to dest_folder."""
    # Clear the destination folder
    """if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)"""
    
    # Move all files from src_folder to dest_folder
    for root, _, files in os.walk(src_folder):
        for file in files:
            full_file_path = os.path.join(root, file)
            shutil.move(full_file_path, dest_folder)


def run_finetune(model_output_path, config_path):
    # Your fine-tuning logic here (e.g., call StyleTTS2 fine-tune script)
    print("ready to finetune:D")
    return 

def update_config(config_path, new_config_path):
    FINAL_CONFIG_PATH = './model/StyleTTS2/Configs/config_ft.yml'
    # Load the existing config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(new_config_path, 'r') as f:
        new_config = yaml.safe_load(f)

    if isinstance(config, dict) and isinstance(new_config, dict):
        config.update(new_config)
    else:
        raise ValueError("Both config and new_config must be dictionaries.")

    #remove existing config_ft.yml file:
    if os.path.exists(FINAL_CONFIG_PATH):
        os.remove(FINAL_CONFIG_PATH)
    
    with open(FINAL_CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)
    

def run_finetune(voice_id):
    global finetune_process
    try:
        # Path to the StyleTTS2 directory where the command should be run
        styletts2_dir = os.path.join(os.getcwd(), "model/StyleTTS2")
        
        # print debug of gpu
        print("GPU is available: ", torch.cuda.is_available())
        
        # Construct the shell command for running the finetuning
        command = [
            "accelerate", "launch",
            "--mixed_precision=fp16",
            "--num_processes=1",
            "train_finetune_accelerate.py",
            "--config_path", "./Configs/config_ft.yml"
        ]
        
        # Start the subprocess and store the process handle
        finetune_process = subprocess.Popen(command, cwd=styletts2_dir)
        print("Model finetuning started successfully")
        
        # Wait for the process to complete
        finetune_process.wait()

        print("calling save_finetuned_model")
        save_finetuned_model(voice_id)

    except Exception as e:
        print(f"Error during model finetuning: {e}")


def find_newest_model(directory):
    # Define the pattern to match the model filenames
    pattern = re.compile(r'epoch_2nd_(\d{5}).pth')

    # List all files in the directory
    files = os.listdir(directory)
    print(f"files: {files}")

    # Filter out the model files and extract their epoch numbers
    model_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            model_files.append((epoch, file))

    # If no model files are found, return None
    if not model_files:
        print("model_files = None")
        return None

    # Find the model file with the highest epoch number
    newest_model = max(model_files, key=lambda x: x[0])[1]

    return os.path.join(directory, newest_model)


def save_finetuned_model(voice_id):
    print("Saving the finetuned model...")
    local_model_dir = './model/StyleTTS2/Models/LJSpeech'
    absolute_model_dir = os.path.abspath(local_model_dir)
    try:
        print(f"Looking for the newest model in: {absolute_model_dir}")
        newest_model_path = find_newest_model(local_model_dir)
    except Exception as e:
        print(f"Error finding the newest model: {e}")
        return None
    
    if newest_model_path:
        print(f"The newest model is: {newest_model_path}")
    else:
        print("No model files found in the directory.")
        return None
    
    # Initialize Firebase Storage
    bucket = storage.bucket()

    # Define destination path in Firebase Storage
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_blob_name = f"finetuned_models/{voice_id}_{timestamp}.tar"
    blob = bucket.blob(destination_blob_name)
    
    
    tar_dir = './finetuned_models' 
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    
    tar_path = f"./finetuned_models/{voice_id}_{timestamp}.tar"

    try:
        with tarfile.open(tar_path, "w") as tar:
            tar.add(newest_model_path, arcname=f"{voice_id}.pth")  # Archive the model file
    except Exception as e:
        print(f"Error creating tar file: {e}")
        return None
    
    # Upload the .tar file to Firebase Storage
    try:
        blob.upload_from_filename(tar_path)
        print(f"Model uploaded successfully to Firebase Storage: {destination_blob_name}")
    except Exception as e:
        print(f"Error uploading the model to Firebase: {e}")
        return None

    return destination_blob_name


def clean_exit():
    clear_directory('temp')
    clear_directory('makeDataset/tools/audio')
    clear_directory('makeDataset/tools/raw_srt')
    clear_directory('makeDataset/tools/srt')
    clear_directory('makeDataset/tools/segmentedAudio')
    clear_directory('makeDataset/tools/trainingdata')
    #clear_directory('trainingdata')
    clear_directory('model/StyleTTS2/Data/wavs')
    clear_directory('model/StyleTTS2/Data')

    
    print("Exiting the application...")
    exit(0)

     

    