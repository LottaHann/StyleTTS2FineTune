import os
import firebase_admin
from firebase_admin import credentials
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    # Base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, 'makeDataset')
    TOOLS_DIR = os.path.join(DATASET_DIR, 'tools')
    DATA_DIR = os.path.join(BASE_DIR, 'Data')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')  # Added this
    
    # Dataset creation directories
    AUDIO_DIR = os.path.join(TOOLS_DIR, 'audio')
    # RAW_SRT_DIR = os.path.join(TOOLS_DIR, 'raw_srt')
    SRT_DIR = os.path.join(TOOLS_DIR, 'srt')
    SEGMENTED_AUDIO_DIR = os.path.join(TOOLS_DIR, 'segmentedAudio')
    TRAINING_DATA_DIR = os.path.join(TOOLS_DIR, 'trainingdata')
    BAD_AUDIO_DIR = os.path.join(TOOLS_DIR, 'badAudio')
    
    # Final dataset directories
    FINAL_WAVS_DIR = os.path.join(DATA_DIR, 'wavs')
    FINAL_RAW_WAVS_DIR = os.path.join(DATA_DIR, 'raw_wavs')
    
    # Configuration files
    DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, 'default_config.yml')
    
    # Firebase configuration
    FIREBASE_CREDENTIALS = 'audiobookgen-firebase-adminsdk-mhp3c-544d551487.json'
    FIREBASE_BUCKET = 'audiobookgen.appspot.com'
    
    # ElevenLabs configuration
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    print(f"ELEVENLABS_API_KEY: {ELEVENLABS_API_KEY}")
    ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
    
    @classmethod
    def initialize_directories(cls):
        """Create all necessary directories if they don't exist"""
        directories = [
            cls.AUDIO_DIR,
            cls.SRT_DIR,
            cls.SEGMENTED_AUDIO_DIR,
            cls.TRAINING_DATA_DIR,
            cls.DATA_DIR,
            cls.FINAL_WAVS_DIR,
            cls.FINAL_RAW_WAVS_DIR,
            cls.TEMP_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def initialize_firebase(cls):
        """Initialize Firebase connection"""
        if not firebase_admin._apps:  # Only initialize if not already initialized
            try:
                cred = credentials.Certificate(cls.FIREBASE_CREDENTIALS)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': cls.FIREBASE_BUCKET
                })
                print("Firebase initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Firebase: {str(e)}")
                # You might want to handle this error differently depending on your needs
                raise 