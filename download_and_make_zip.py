import requests
import json
from elevenlabs import ElevenLabs
from training_texts import get_dataset
from training_dialogs import get_dialog_array
import os

def get_all_history_items(api_key, voice_id):
    headers = {
        'xi-api-key': api_key
    }
    
    all_items = []
    has_more = True
    start_after_id = None
    
    while has_more:
        # Construct URL with parameters
        url = f'https://api.elevenlabs.io/v1/history?page_size=1000&voice_id={voice_id}'
        if start_after_id:
            url += f'&start_after_history_item_id={start_after_id}'
            
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching history: {response.status_code}")
            break
            
        data = response.json()
        history_items = data.get('history', [])
        all_items.extend(history_items)
        
        has_more = data.get('has_more', False)
        if has_more and history_items:
            start_after_id = history_items[-1]['history_item_id']
            

    print(f"Found {len(all_items)} history items")
    return all_items

def download_matching_audio_files(voice_id, api_key):
    # Get all history items using pagination
    history_items = get_all_history_items(api_key, voice_id)
    
    # Create tuple array of (history_item_id, text)
    voice_history = [(item['history_item_id'], item['text']) 
                    for item in history_items]
    
    # Initialize ElevenLabs client
    client = ElevenLabs(api_key=api_key)
    
    # Get training texts
    training_texts_1 = get_dataset(1)
    training_texts_2 = get_dialog_array(1)
    training_texts = training_texts_1 + training_texts_2
    
    # Create output directory if it doesn't exist
    os.makedirs("output_wavs", exist_ok=True)
    
    matches_found = 0
    
    # Process each training text
    for index, training_text in enumerate(training_texts, 1):
        # Find matches in history
        matches = [(id, text) for id, text in voice_history if text == training_text]
        
        if len(matches) > 1:
            print(f"Warning: Multiple matches found for text {index}. Skipping.")
            continue
        
        if len(matches) == 0:
            print(f"No match found for text {index}. Skipping.")
            continue
        
        # Download the matching audio file
        history_item_id = matches[0][0]
        
        headers = {
            'Content-Type': 'application/json',
            'xi-api-key': api_key
        }
        
        data = {
            'history_item_ids': [history_item_id],
            'output_format': 'wav'
        }
        
        response = requests.post(
            'https://api.elevenlabs.io/v1/history/download',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            with open(f"output_wavs/{index}.wav", 'wb') as f:
                f.write(response.content)
            matches_found += 1
            print(f"Successfully downloaded {index}.wav")
        else:
            print(f"Failed to download audio for text {index}")
    
    print(f"\nDownload complete. Found and downloaded {matches_found} matching audio files.")

if __name__ == "__main__":
    VOICE_ID = "l26AVwIi7QuBeQguFlXa"
    API_KEY = "sk_b94ebb4bb50dc4e8de663c1958d5fc656a44eb90872df536"
    download_matching_audio_files(VOICE_ID, API_KEY)
