from makeDataset.tools.format_srt import format_srt_file
from makeDataset.tools.transcribe_audio import transcribe_all_files
from makeDataset.tools.phonemized_func import phonemize_transcriptions
from makeDataset.tools.srtsegmenter_func import process_audio_segments
import argparse
import shutil
from datetime import datetime



def makedataset():
    audio_dir = "Data/wavs"
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
    shutil.copytree('./makeDataset/tools/trainingdata', f'./trainingdata{datetime.now().strftime("%Y%m%d%H%M%S")}')

    #copy segmented audio files from makeDataset/tools/segmentedAudio to segmentedAudio
    shutil.copytree('./makeDataset/tools/segmentedAudio', f'./segmentedAudio{datetime.now().strftime("%Y%m%d%H%M%S")}')

    

if __name__ == "__main__":
    makedataset()
    




