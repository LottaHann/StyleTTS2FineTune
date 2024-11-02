import pysrt
from pydub import AudioSegment
import os
from tqdm import tqdm
import glob

def process_audio_segments(buffer_time=200, max_allowed_gap_multiplier=1.5, min_duration=1850, max_duration=12000):
    """
    Processes audio files based on subtitles in SRT files, segments them, and saves them in separate directories.
    
    Parameters:
    - buffer_time: int, optional
        Time in milliseconds to add as a buffer around each subtitle.
    - max_allowed_gap_multiplier: float, optional
        Multiplier for buffer time to define the maximum allowed gap between subtitles.
    - min_duration: int, optional
        Minimum duration (in ms) for a segment to be considered valid.
    - max_duration: int, optional
        Maximum duration (in ms) for a segment to be considered valid.
    """

    # Directories
    output_dir = './makeDataset/tools/segmentedAudio/'
    bad_audio_dir = './makeDataset/tools/badAudio/'
    srt_dir = './makeDataset/tools/srt/'
    audio_dir = './makeDataset/tools/audio/'
    training_data_dir = './makeDataset/tools/trainingdata'
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bad_audio_dir, exist_ok=True)
    os.makedirs(srt_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(training_data_dir, exist_ok=True)

    # Get lists of srt and audio files
    srt_list = glob.glob(os.path.join(srt_dir, "*.srt"))
    audio_list = glob.glob(os.path.join(audio_dir, "*.wav"))

    # Check if there are files to process
    if len(srt_list) == 0 or len(audio_list) == 0:
        raise Exception(f"You need to have at least 1 srt file and 1 audio file, you have {len(srt_list)} srt and {len(audio_list)} audio files!")

    # Print count of SRT files
    print(f"SRT Files: {len(srt_list)}")
    
    max_allowed_gap = max_allowed_gap_multiplier * buffer_time  # Calculate max allowed gap

    def convert_srt_time_to_milliseconds(srt_time):
        """Convert SRT time format to milliseconds."""
        hours, minutes, seconds_milliseconds = srt_time.split(':')
        seconds, milliseconds = seconds_milliseconds.split(',')
        return (int(hours) * 3600 * 1000 + int(minutes) * 60 * 1000 + int(seconds) * 1000 + int(milliseconds))

    # Process each SRT file
    for sub_file in tqdm(srt_list, desc="Processing SRT Files"):
        subs = pysrt.open(sub_file)
        audio_name = os.path.basename(sub_file).replace(".srt", ".wav")
        audio = AudioSegment.from_wav(os.path.join(audio_dir, audio_name))



        with open(os.path.join(training_data_dir, 'output.txt'), 'a+') as out_file:
            for i, sub in enumerate(subs):
                start_time = convert_srt_time_to_milliseconds(str(sub.start))
                end_time = convert_srt_time_to_milliseconds(str(sub.end))

                if i < len(subs) - 1:
                    next_sub = subs[i + 1]
                    next_start_time = convert_srt_time_to_milliseconds(str(next_sub.start))
                    gap_to_next = next_start_time - end_time

                    if gap_to_next > max_allowed_gap:
                        end_time += buffer_time
                    else:
                        adjustment = min(buffer_time, gap_to_next // 2)
                        end_time += adjustment
                else:
                    end_time += buffer_time

                audio_segment = audio[start_time:end_time]
                duration = len(audio_segment)
                filename = f'{audio_name[:-4]}_{i}.wav'

                if min_duration <= duration <= max_duration:
                    audio_segment.export(os.path.join(output_dir, filename), format='wav')
                    out_file.write(f'{filename}|{sub.text}|1\n')
                else:
                    audio_segment.export(os.path.join(bad_audio_dir, filename), format='wav')
