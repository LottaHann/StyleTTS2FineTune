import pysrt
from pydub import AudioSegment
import os
from tqdm import tqdm
import glob
import math

def is_silent_rms(audio_chunk, rms_thresh=120):
    """Check if an audio chunk is silent using direct RMS threshold"""
    return audio_chunk.rms < rms_thresh

def clean_segment(audio):
    """Remove artifacts at the end if last 400ms starts silent but isn't completely silent"""
    if len(audio) < 400:
        return audio
        
    last_part = audio[-400:]
    first_10ms = last_part[:10]
    print(f"DEBUG: Segment length: {len(audio)}ms")
    print(f"DEBUG: First 10ms RMS: {first_10ms.rms}")
    print(f"DEBUG: Last 400ms RMS: {last_part.rms}")
    
    # Using RMS values directly: first_10ms should be very quiet (< 100)
    # but last_part should have some noticeable sound (> 1)
    if is_silent_rms(first_10ms, 100) and last_part.rms > 1:
        print("DEBUG: Removing last 400ms")
        return audio[:-400]
    else:
        print("DEBUG: Keeping full segment")
    
    return audio

def add_silence_buffers(audio, target_silence=200):
    """Add silence to ensure target_silence ms of silence at start and end"""
    # Add a small crossfade to prevent abrupt cuts
    crossfade_duration = 50  # ms
    
    # Find first non-silent part with lower threshold
    for i in range(0, len(audio)-10, 10):
        if not is_silent_rms(audio[i:i+10], 120):
            silence_start = i
            break
    else:
        silence_start = 0

    # Find last non-silent part with lower threshold
    for i in range(len(audio)-10, 0, -10):
        if not is_silent_rms(audio[i:i+10], 120):
            silence_end = len(audio) - i
            break
    else:
        silence_end = 0

    # Calculate needed silence with a bit more buffer
    start_buffer = max(0, target_silence - silence_start) + 50
    end_buffer = max(0, target_silence - silence_end) + 50

    result = (AudioSegment.silent(duration=start_buffer) + 
             audio + 
             AudioSegment.silent(duration=end_buffer))
    
    # Apply gentler crossfade at the boundaries
    if len(result) > crossfade_duration * 2:
        result = result.fade_in(crossfade_duration).fade_out(crossfade_duration)
    
    return result

def optimize_silence(audio, min_silence_len=200, max_silence_len=500, step=10):
    """
    Optimize silence segments in the middle of the audio file.
    Ignores buffer silences at start and end.
    Returns tuple of (processed_audio, original_duration, final_duration)
    """
    original_duration = len(audio)
    
    # Skip the first and last 200ms (buffer zones)
    working_audio = audio[200:-200]
    
    # Find silence segments
    silence_segments = []
    start = None
    
    # Scan for silence segments
    for i in range(0, len(working_audio)-step, step):
        chunk = working_audio[i:i+step]
        is_silent = is_silent_rms(chunk)
        
        if is_silent and start is None:
            start = i
        elif not is_silent and start is not None:
            silence_segments.append((start, i))
            start = None
    
    # Add last segment if ends with silence
    if start is not None:
        silence_segments.append((start, len(working_audio)))
    
    # Process each silence segment
    result_audio = audio[:200]  # Start with initial buffer
    last_end = 0
    
    total_silence_removed = 0
    
    for start, end in silence_segments:
        # Add audio before silence
        result_audio += working_audio[last_end:start]
        
        # Calculate silence duration
        silence_duration = end - start
        
        # Apply silence optimization rules
        if silence_duration > max_silence_len:
            # If longer than max_silence_len, reduce to max_silence_len
            new_silence = AudioSegment.silent(duration=max_silence_len)
            total_silence_removed += silence_duration - max_silence_len
        elif silence_duration > min_silence_len:
            # If between min and max, reduce to 60%
            new_duration = int(silence_duration * 0.6)
            total_silence_removed += silence_duration - new_duration
            new_silence = AudioSegment.silent(duration=new_duration)
        else:
            # If shorter than min_silence_len, keep as is
            new_silence = working_audio[start:end]
        
        result_audio += new_silence
        last_end = end

    # Add remaining audio and final buffer
    result_audio += working_audio[last_end:] + audio[-200:]
    
    final_duration = len(result_audio)
    
    return result_audio, original_duration, final_duration

def process_audio_segments(buffer_time=200, min_duration=1850, max_duration=8000):
    """Main processing function"""
    print("Processing audio segments...")
    # Setup directories
    dirs = {
        'output': './makeDataset/tools/segmentedAudio/',
        'bad': './makeDataset/tools/badAudio/',
        'srt': './makeDataset/tools/srt/',
        'audio': './makeDataset/tools/audio/',
        'training': './makeDataset/tools/trainingdata'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Add tracking for input files
    print("\nInput File Analysis:")
    audio_files = os.listdir(dirs['audio'])
    srt_files = glob.glob(os.path.join(dirs['srt'], "*.srt"))
    print(f"Found {len(audio_files)} audio files: {audio_files}")
    print(f"Found {len(srt_files)} SRT files: {[os.path.basename(f) for f in srt_files]}")

    if not srt_files:
        raise Exception("No SRT files found!")

    # Clear output file
    with open(os.path.join(dirs['training'], 'output.txt'), 'w') as _:
        pass


    # first print the total length of all the audio files (in minutes)
    total_length = sum(len(AudioSegment.from_wav(os.path.join(dirs['audio'], audio_name))) for audio_name in os.listdir(dirs['audio']))
    print(f"Total length of all audio files: {total_length/1000/60} minutes")

    total_segment_count = 0 
    total_duration = 0

    # Add counters for tracking
    stats = {
        'total_segments_processed': 0,
        'segments_too_short': 0,
        'segments_too_long': 0,
        'segments_kept': 0
    }

    # Process each SRT file
    for srt_file in tqdm(srt_files, desc="Processing SRT Files"):
        print(f"\nProcessing SRT file: {srt_file}")
        subs = pysrt.open(srt_file)
        audio_name = os.path.basename(srt_file).replace(".srt", ".wav")
        
        if not os.path.exists(os.path.join(dirs['audio'], audio_name)):
            print(f"WARNING: Missing audio file for {audio_name}")
            continue

        audio = AudioSegment.from_wav(os.path.join(dirs['audio'], audio_name))
        
        # Reset segment counter for each file
        file_segment_count = 0
        
        # Process segments
        current_segment = None
        
        with open(os.path.join(dirs['training'], 'output.txt'), 'a') as out_file:
            for i, sub in enumerate(subs):
                # Calculate cut points
                current_end = sub.end.ordinal
                next_start = subs[i + 1].start.ordinal if i < len(subs) - 1 else None
                
                # Calculate the start point (using previous midpoint if available)
                if i > 0:
                    prev_end = subs[i-1].end.ordinal
                    start_point = prev_end + (sub.start.ordinal - prev_end) // 2
                else:
                    start_point = sub.start.ordinal
                
                # Calculate the end point (using next midpoint if available)
                if next_start is not None:
                    end_point = current_end + (next_start - current_end) // 2
                else:
                    end_point = current_end

                # Extract and clean segment using the new points
                segment = audio[start_point:end_point]
                print(f"\nSubtitle {i+1}/{len(subs)}:")
                print(f"Original timing: {sub.start} -> {sub.end}")
                print(f"Modified timing: {start_point}ms -> {end_point}ms")
                print(f"Text: {sub.text.strip()}")
                
                print(f"Segment duration before cleaning: {len(segment)}ms")
                cleaned = clean_segment(segment)
                print(f"Segment duration after cleaning: {len(cleaned)}ms")
                
                if current_segment is None:
                    current_segment = {
                        'audio': cleaned,
                        'text': sub.text.strip(),
                        'start': start_point
                    }
                    continue
                
                # Try to merge with previous segment
                potential_duration = len(current_segment['audio']) + len(cleaned)
                
                if potential_duration <= max_duration:
                    print(f"Merging segments: {current_segment['text']} + {sub.text.strip()}")
                    current_segment['audio'] += cleaned
                    current_segment['text'] = f"{current_segment['text']} {sub.text.strip()}"
                else:
                    print(f"Starting new segment due to duration limit")
                    # Process current segment before starting new one
                    if current_segment:
                        audio_out = current_segment['audio']
                        stats['total_segments_processed'] += 1
                        
                        # Add silence buffers
                        audio_out = add_silence_buffers(audio_out)
                        
                        # Optimize silence segments
                        audio_out, original_duration, final_duration = optimize_silence(audio_out)
                        
                        # Save if duration is valid
                        duration = final_duration
                        filename = f'{audio_name[:-4]}_{file_segment_count}.wav'
                        
                        if duration < min_duration:
                            print(f"Skipping {filename}: Too short ({duration}ms < {min_duration}ms)")
                            stats['segments_too_short'] += 1
                            audio_out.export(os.path.join(dirs['bad'], filename), format='wav')
                        elif duration > max_duration:
                            print(f"Skipping {filename}: Too long ({duration}ms > {max_duration}ms)")
                            stats['segments_too_long'] += 1
                            audio_out.export(os.path.join(dirs['bad'], filename), format='wav')
                        else:
                            stats['segments_kept'] += 1
                            audio_out.export(os.path.join(dirs['output'], filename), format='wav')
                            out_file.write(f'{filename}|{current_segment["text"]}|1\n')
                            print(f"  Saving segment {filename} with text: {current_segment['text']}")
                            total_duration += duration
                            file_segment_count += 1
                            total_segment_count += 1
                    
                    # Start new segment with current subtitle
                    current_segment = {
                        'audio': cleaned,
                        'text': sub.text.strip(),
                        'start': start_point
                    }
            
            # Process the last segment if it exists
            if current_segment:
                audio_out = current_segment['audio']
                stats['total_segments_processed'] += 1
                
                audio_out = add_silence_buffers(audio_out)
                audio_out, original_duration, final_duration = optimize_silence(audio_out)
                
                duration = final_duration
                filename = f'{audio_name[:-4]}_{file_segment_count}.wav'
                
                if duration < min_duration:
                    print(f"Skipping final {filename}: Too short ({duration}ms < {min_duration}ms)")
                    stats['segments_too_short'] += 1
                    audio_out.export(os.path.join(dirs['bad'], filename), format='wav')
                elif duration > max_duration:
                    print(f"Skipping final {filename}: Too long ({duration}ms > {max_duration}ms)")
                    stats['segments_too_long'] += 1
                    audio_out.export(os.path.join(dirs['bad'], filename), format='wav')
                else:
                    stats['segments_kept'] += 1
                    audio_out.export(os.path.join(dirs['output'], filename), format='wav')
                    out_file.write(f'{filename}|{current_segment["text"]}|1\n')
                    print(f"  Saving final segment {filename} with text: {current_segment['text']}")
                    total_duration += duration
                    file_segment_count += 1
                    total_segment_count += 1

    print(f"\nDetailed Processing Summary:")
    print(f"Total segments processed: {stats['total_segments_processed']}")
    print(f"Segments too short (<{min_duration}ms): {stats['segments_too_short']}")
    print(f"Segments too long (>{max_duration}ms): {stats['segments_too_long']}")
    print(f"Segments kept: {stats['segments_kept']} ({(stats['segments_kept']/stats['total_segments_processed']*100):.1f}% of total)")
    print(f"Total duration of kept segments: {total_duration/1000:.2f} seconds")
    print(f"Average duration of kept segments: {total_duration/total_segment_count:.2f}ms")
    
    return total_segment_count, total_duration

if __name__ == "__main__":
    test_file = "./raw_3.wav"
    print(f"Testing processing on {test_file}")
    
    # Load full audio
    audio = AudioSegment.from_wav(test_file)
    print(f"Full audio duration: {len(audio)}ms")
    
    # Segment 1: 0ms -> 5040ms
    print("\nProcessing Segment 1:")
    segment1 = audio[0:5040]
    print(f"Original duration: {len(segment1)}ms")
    
    cleaned1 = clean_segment(segment1)
    buffered1 = add_silence_buffers(cleaned1, target_silence=200)
    optimized1, orig_dur1, final_dur1 = optimize_silence(buffered1)
    optimized1.export("./raw_3_optimized1.wav", format="wav")
    buffered1.export("./raw_3_buffered1.wav", format="wav")
    cleaned1.export("./raw_3_cleaned1.wav", format="wav")