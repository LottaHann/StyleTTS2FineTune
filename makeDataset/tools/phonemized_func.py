
from phonemizer import phonemize
import os
from tqdm import tqdm

def phonemize_transcriptions(input_file, train_output_file, val_output_file, language="en-us"):
    """Phonemize transcriptions and split into training and validation sets.

    Args:
        input_file (str): Path to the input file containing transcriptions.
        train_output_file (str): Path to save the training list.
        val_output_file (str): Path to save the validation list.
        language (str): The language for phonemization (default: "en-us").
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Prepare lists to hold filenames, transcriptions, and speakers
    filenames = []
    transcriptions = []
    speakers = []
    phonemized_lines = []

    for line in lines:
        filename, transcription, speaker = line.strip().split("|")
        filenames.append(filename)
        transcriptions.append(transcription)
        speakers.append(speaker)

    # Phonemize all text in one go to avoid memory errors
    phonemized = phonemize(
        transcriptions,
        language=language,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
    )

    for i in tqdm(range(len(filenames))):
        phonemized_lines.append(
            (filenames[i], f"{filenames[i]}|{phonemized[i]}|{speakers[i]}\n")
        )

    phonemized_lines.sort(key=lambda x: int(x[0].split("_")[1].split(".")[0]))

    # Split into training and validation sets
    train_lines = phonemized_lines[:int(len(phonemized_lines) * 0.9)]
    val_lines = phonemized_lines[int(len(phonemized_lines) * 0.9):]

    with open(train_output_file, "w+", encoding="utf-8") as f:
        for _, line in train_lines:
            f.write(line)

    with open(val_output_file, "w+", encoding="utf-8") as f:
        for _, line in val_lines:
            f.write(line)


    
