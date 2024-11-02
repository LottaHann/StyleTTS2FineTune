import os
import re
from pydub import AudioSegment

def convert_srt_time_to_milliseconds(srt_time):
    """Convert SRT time format to milliseconds."""
    hours, minutes, seconds = map(float, re.split('[:,]', srt_time))
    return int((hours * 3600 + minutes * 60 + seconds) * 1000)

def segment_audio_from_srt(audio_folder, srt_folder, output_folder):
    """Segment audio files based on SRT files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each SRT file
    for srt_file in os.listdir(srt_folder):
        if srt_file.endswith('.srt'):
            base_name = os.path.splitext(srt_file)[0]
            audio_file_path = os.path.join(audio_folder, f"{base_name}.wav")
            audio = AudioSegment.from_wav(audio_file_path)
            
            with open(os.path.join(srt_folder, srt_file), 'r', encoding='utf-8') as f:
                lines = f.readlines()

            segment_count = 0
            start_time = None
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+$', line):  # Line with segment number
                    continue
                if '-->' in line:  # Line with timestamps
                    start, end = line.split(' --> ')
                    if start_time is None:
                        start_time = convert_srt_time_to_milliseconds(start)
                    end_time = convert_srt_time_to_milliseconds(end)
                    
                    # Create segment
                    segment = audio[start_time:end_time]
                    segment_count += 1
                    segment_file_path = os.path.join(output_folder, f"{base_name}_segment{segment_count}.wav")
                    segment.export(segment_file_path, format="wav")
                    
                    start_time = None  # Reset start time for the next segment

            print(f"Processed {srt_file}: {segment_count} segments created.")

# Usage
if __name__ == "__main__":
    audio_folder = "audio"
    srt_folder = "srt"
    output_folder = "segmentedAudio"
    segment_audio_from_srt(audio_folder, srt_folder, output_folder)
