import os
import glob
from tqdm import tqdm
import pysrt
import firebase_admin
from firebase_admin import firestore


def parse_time(time_str):
    """Parse a time string from SRT format (HH:MM:SS,mmm) to milliseconds."""
    hours, minutes, seconds_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_milliseconds.split(',')
    total_milliseconds = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)
    return total_milliseconds

def check_last_character(text):
    """Check if the last character of the text is a complete sentence."""
    return text.endswith(('.', '!', '?'))

def combine_texts(text, next_text):
    return text + " " + next_text

def format_srt_file(srt_file_path):
    print("Formatting SRT files...")
    """Format the SRT file so that each segment ends with a complete sentence."""
    srt_list = glob.glob(srt_file_path)

    for sub_file in tqdm(srt_list):  # Iterate over all srt files
        subs = pysrt.open(sub_file)
        
        i = 0
        while i < len(subs):
            current_sub = subs[i]
            current_text = current_sub.text.strip()

            # Combine subtitles if the current text does not end with a complete sentence
            while not check_last_character(current_text):
                # Prevent out-of-bounds access
                if i + 1 >= len(subs):
                    break
                
                next_sub = subs[i + 1]
                next_text = next_sub.text.strip()

                # Combine current and next subtitles
                current_text = combine_texts(current_text, next_text)
                current_sub.text = current_text
                current_sub.end = next_sub.end  # Update end time to the next subtitle's end time

                # Remove the next subtitle and do not increment i since we merged it
                subs.remove(next_sub)

            # Move to the next subtitle
            i += 1

        # Save the modified subtitles back to the output SRT file
        output_file_name = f"./makeDataset/tools/srt/{os.path.basename(sub_file)}"  # Define output path
        subs.save(output_file_name, encoding='utf-8')


"""
if __name__ == "__main__":
    # Example usage
    #os.chdir('./makeDataset/tools')
    raw_srt_folder = "raw_srt/*.srt"
    output_srt_folder = "srt"

    if not os.path.exists(output_srt_folder):
        os.makedirs(output_srt_folder)

    format_srt_file(raw_srt_folder, output_srt_folder)

    print("SRT files have been formatted and saved to the 'srt' folder.")
"""
