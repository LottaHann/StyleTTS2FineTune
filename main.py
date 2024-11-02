from makeDataset.tools.format_srt import format_srt_file
from makeDataset.tools.transcribe_audio import transcribe_all_files


def main():
    audio_dir = "Data/wavs"
    print("Transcribing audio files...")
    transcribe_all_files(audio_dir)

    print("raw srt files have been created")

    srt_file_path = "./makeDataset/tools/raw_srt/*.srt"

    format_srt_file(srt_file_path)

    print("srt files are being formatted")

if __name__ == "__main__":
    main()




