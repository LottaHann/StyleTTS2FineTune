from flask import Flask, request, jsonify
#from makedataset import makedataset
import yaml
from app_func import makedataset, run_finetune, download_file, clear_directory, extract_zip, copy_files, update_config, run_finetune,  move_files, clean_exit, save_finetuned_model, finetune_process
import shutil
import os
from download_model import download_model

import firebase_admin
from firebase_admin import credentials, storage

cred = credentials.Certificate('audiobookgen-firebase-adminsdk-mhp3c-544d551487.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'audiobookgen.appspot.com'
})

app = Flask(__name__)

AUDIO_DIR = "./makeDataset/tools/audio"
DEFAULT_CONFIG_PATH = './default_config.yml'
FINAL_CONFIG_PATH = './model/StyleTTS2/Configs/config_ft.yml'

@app.route('/finetune', methods=['POST'])
def finetune():
    data = request.json
    voice_id = data.get('voice_id')
    audio_zip_url = data.get('audio_zip_url')
    config_url = data.get('config_url')

    if not voice_id or not audio_zip_url:
        return jsonify({"error": "Missing voice_id or audio_zip_url"}), 400

    # Step 1: Download the audio zip
    audio_zip_path = f"./temp/{voice_id}_audio.zip"
    try:
        download_file(audio_zip_url, audio_zip_path)
    except Exception as e:
        clean_exit()
        return jsonify({"error downloading audio files": str(e)}), 500
    
        
    # Step 2: Clear the audio directory and extract the new audio files
    try:
        clear_directory(AUDIO_DIR) 
        extract_zip(audio_zip_path, AUDIO_DIR)
    except Exception as e:
        clean_exit()
        return jsonify({"error extracting audio files": str(e)}), 500
    
    makedataset()

    wav_dir_for_finetuning = './model/StyleTTS2/Data/wavs'


    clear_directory('model/StyleTTS2/Data')
    
    #create dir if it doesnt exist:
    if not os.path.exists(wav_dir_for_finetuning):
        os.makedirs(wav_dir_for_finetuning)
        print("created wavs dir")
    else:
        clear_directory(wav_dir_for_finetuning)

    try:
        move_files('makeDataset/tools/segmentedAudio', wav_dir_for_finetuning)
        move_files('makeDataset/tools/trainingdata', 'model/StyleTTS2/Data')
        print("moved files")
    except Exception as e:
        clean_exit()        
        return jsonify({"error moving audio files": str(e)}), 500
    

    shutil.copy('./OOD_texts.txt', './model/StyleTTS2/Data')

    
    custom_config_path = './temp/custom_config.yml'
    if config_url:
        download_file(config_url, custom_config_path)
        update_config(DEFAULT_CONFIG_PATH, custom_config_path)
    else:
        shutil.copy(DEFAULT_CONFIG_PATH, FINAL_CONFIG_PATH)

    download_model()
        
    run_finetune(voice_id)

    save_finetuned_model(voice_id)

    clear_directory('temp')

    return jsonify({"message": f"Model for {voice_id} trained successfully"}), 200


@app.route('/stop_finetune', methods=['POST'])
def stop_finetune():
    global finetune_process
    if finetune_process is not None:
        finetune_process.terminate()  # Terminate the process
        finetune_process = None  # Reset the process variable
        return jsonify({"message": "Finetuning process stopped"}), 200
    else:
        return jsonify({"error": "No finetuning process is running"}), 400


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=False, host='0.0.0.0', port=5000)
