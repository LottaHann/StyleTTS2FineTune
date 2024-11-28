import os
import librosa
import soundfile as sf
from pathlib import Path

def resample_audio(input_path, output_path, target_sr):
    try:
        print(f"Resampling {input_path}...")
        audio, sr = librosa.load(input_path, sr=None)
        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        sf.write(output_path, audio, target_sr)
        print(f"Successfully resampled to {output_path}")
    except Exception as e:
        print(f"Error resampling audio {input_path}: {str(e)}")
        return False
    return True