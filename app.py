from flask import Flask, request, jsonify
import os
import requests
import zipfile
import shutil
from makedataset import makedataset

app = Flask(__name__)

# Path to the directory where audio files will be saved
AUDIO_DIR = "./makeDataset/tools/audio"

@app.route('/finetune', methods=['POST'])
def finetune():
    data = request.json
    voice_id = data.get('voice_id')
    audio_zip_url = data.get('audio_zip_url')
    config_url = data.get('config_url')

    if not voice_id or not audio_zip_url:
        return jsonify({"error": "Missing voice_id or audio_zip_url"}), 400

    # Step 1: Download the audio zip
    audio_zip_path = f"./audio_zips/{voice_id}_audio.zip"
    try:
        download_file(audio_zip_url, audio_zip_path)
    except Exception as e:
        return jsonify({"error downloading audio files": str(e)}), 500
    
    # Step 2: Clear the audio directory and extract the new audio files
    try:
        clear_directory(AUDIO_DIR)  # Clear previous audio files
        extract_zip(audio_zip_path, AUDIO_DIR)  # Extract new files into the audio directory
        #copy_audio_files(AUDIO_DIR, './makeDataset/tools/audio')
    except Exception as e:
        return jsonify({"error extracting audio files": str(e)}), 500
    
    # Step 3: Download and process the config file
    config_path = f"./config/{voice_id}_config.yml"
    if config_url:
        download_file(config_url, config_path)
    #else:
        # Use default config if not provided
        #shutil.copy('./default_config.yml', config_path)

    # Step 4: Call dataset creation and finetuning
    makedataset()
    run_finetune(f"./models/{voice_id}.pth", config_path)

    return jsonify({"message": f"Model for {voice_id} trained successfully"}), 200


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

def copy_audio_files(src_folder, dest_folder):
    """Copy audio files from src_folder to dest_folder."""
    # Clear the destination folder
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)
    
    # Copy only audio files from src_folder to dest_folder
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.wav'):  # Assuming the audio files are .wav format
                full_file_path = os.path.join(root, file)
                shutil.copy(full_file_path, dest_folder)

def run_finetune(model_output_path, config_path):
    # Your fine-tuning logic here (e.g., call StyleTTS2 fine-tune script)
    print("ready to finetune:D")
    return 

if __name__ == '__main__':
    app.run(debug=True)
