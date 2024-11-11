import os
import requests

def download_model():
    # URL of the model file on Hugging Face
    url = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth"
    
    # Target path to save the model file
    target_dir = "model/StyleTTS2/Models/LibriTTS"
    target_path = os.path.join(target_dir, "epochs_2nd_00020.pth")

    if os.path.exists(target_path):
        print(f"Model file already exists at {target_path}")
        return

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Download the file
    print("Downloading model file...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Model downloaded successfully to {target_path}")
    else:
        print(f"Failed to download model. Status code: {response.status_code}")

# Call the function to download the model
if __name__ == "__main__":
    download_model()
