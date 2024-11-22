
from phonemizer import phonemize
import os
from tqdm import tqdm

def phonemize_transcriptions(input_file, train_output_file, val_output_file, language="en-us"):
    """Phonemize transcriptions and split into training and validation sets."""
    print(f"\nStarting phonemization process...")
    print(f"Input file: {input_file}")
    print(f"Train output: {train_output_file}")
    print(f"Val output: {val_output_file}")
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        with open(input_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Read {len(lines)} lines from input file")

        # Prepare lists to hold filenames, transcriptions, and speakers
        filenames = []
        transcriptions = []
        speakers = []
        phonemized_lines = []

        # Parse input lines
        print("Parsing input lines...")
        for i, line in enumerate(lines, 1):
            try:
                parts = line.strip().split("|")
                if len(parts) != 3:
                    print(f"Warning: Line {i} has incorrect format: {line.strip()}")
                    continue
                    
                filename, transcription, speaker = parts
                filenames.append(filename)
                transcriptions.append(transcription)
                speakers.append(speaker)
            except Exception as e:
                print(f"Error parsing line {i}: {str(e)}")
                print(f"Line content: {line.strip()}")
                continue

        print(f"Successfully parsed {len(filenames)} valid entries")

        # Phonemize all text
        print("\nPerforming phonemization...")
        try:
            phonemized = phonemize(
                transcriptions,
                language=language,
                backend="espeak",
                preserve_punctuation=True,
                with_stress=True,
            )
            print(f"Phonemization completed successfully")
        except Exception as e:
            print(f"Error during phonemization: {str(e)}")
            raise

        # Create phonemized lines with progress bar
        print("\nProcessing phonemized results...")
        for i in tqdm(range(len(filenames)), desc="Creating phonemized lines"):
            try:
                phonemized_lines.append(
                    (filenames[i], f"{filenames[i]}|{phonemized[i]}|{speakers[i]}\n")
                )
            except Exception as e:
                print(f"Error processing entry {i}: {str(e)}")
                continue

        # Sort lines
        print("\nSorting entries...")
        try:
            phonemized_lines.sort(key=lambda x: int(x[0].split("_")[1].split(".")[0]))
        except Exception as e:
            print(f"Warning: Error during sorting: {str(e)}")
            print("Continuing with unsorted lines...")

        # Split into training and validation sets
        split_idx = int(len(phonemized_lines) * 0.9)
        train_lines = phonemized_lines[:split_idx]
        val_lines = phonemized_lines[split_idx:]

        print(f"\nSplit distribution:")
        print(f"Training set: {len(train_lines)} entries")
        print(f"Validation set: {len(val_lines)} entries")

        # Ensure output directories exist
        os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
        os.makedirs(os.path.dirname(val_output_file), exist_ok=True)

        # Write output files
        print("\nWriting output files...")
        try:
            with open(train_output_file, "w+", encoding="utf-8") as f:
                for _, line in train_lines:
                    f.write(line)
            print(f"Training file written: {train_output_file}")

            with open(val_output_file, "w+", encoding="utf-8") as f:
                for _, line in val_lines:
                    f.write(line)
            print(f"Validation file written: {val_output_file}")

        except Exception as e:
            print(f"Error writing output files: {str(e)}")
            raise

        print("\nPhonemization process completed successfully!")
        return True

    except Exception as e:
        print(f"\nFatal error during phonemization process: {str(e)}")
        return False


    
