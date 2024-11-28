import pysrt
from pydub import AudioSegment
import os
from tqdm import tqdm
import glob
import math
from pydub.utils import make_chunks
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import io
from config import Config

def is_silent_rms(audio_chunk, rms_thresh=120):
    """Check if an audio chunk is silent using direct RMS threshold"""
    return audio_chunk.rms < rms_thresh

def clean_segment(audio):
    """Remove artifacts at the end if last 400ms starts silent but isn't completely silent"""
    if len(audio) < 400:
        return audio
        
    last_part = audio[-400:]
    first_10ms = last_part[:10]
    
    # Using RMS values directly: first_10ms should be very quiet (< 100)
    # but last_part should have some noticeable sound (> 1)
    if is_silent_rms(first_10ms, 100) and last_part.rms > 1:
        return audio[:-400]
    
    return audio

def add_silence_buffers(audio, target_silence=200):
    """Standardize silence to exactly target_silence ms at start and end"""    
    # Find first non-silent part
    for i in range(0, len(audio)-10, 10):
        if not is_silent_rms(audio[i:i+10], 120):
            silence_start = i
            break
    else:
        silence_start = 0

    # Find last non-silent part
    for i in range(len(audio)-10, 0, -10):
        if not is_silent_rms(audio[i:i+10], 120):
            silence_end = len(audio) - i
            break
    else:
        silence_end = 0

    # Trim or add silence at start
    if silence_start > target_silence:
        # If too much silence, trim from left
        audio = audio[(silence_start - target_silence):]
    elif silence_start < target_silence:
        # If too little silence, add more
        audio = AudioSegment.silent(duration=target_silence - silence_start) + audio

    # Trim or add silence at end
    if silence_end > target_silence:
        # If too much silence, trim from right
        audio = audio[:-1 * (silence_end - target_silence)]
    elif silence_end < target_silence:
        # If too little silence, add more
        audio = audio + AudioSegment.silent(duration=target_silence - silence_end)
    
    return audio

def optimize_silence(audio, min_silence_len=200, max_silence_len=500, step=10):
    """
    Optimize silence segments in the middle of the audio file.
    Returns tuple of (processed_audio, original_duration, final_duration)
    """
    original_duration = len(audio)
    
    # Skip the first and last 200ms (buffer zones)
    working_audio = audio[200:-200]
    
    # Find silence segments using a more robust detection method
    silence_segments = []
    start = None
    consecutive_silent = 0
    required_silent_chunks = 3  # Require multiple consecutive silent chunks
    
    # Scan for silence segments with improved detection
    for i in range(0, len(working_audio)-step, step):
        chunk = working_audio[i:i+step]
        is_silent = is_silent_rms(chunk, rms_thresh=110)  # Slightly more sensitive threshold
        
        if is_silent:
            consecutive_silent += 1
            if start is None and consecutive_silent >= required_silent_chunks:
                start = i - (step * (required_silent_chunks-1))
        else:
            consecutive_silent = 0
            if start is not None:
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

def calculate_lufs(audio, sample_rate=22050):
    """Calculate LUFS using pyloudnorm with proper channel handling"""
    try:
        # Convert to mono if stereo
        if audio.channels == 2:
            audio = audio.set_channels(1)
        
        # Convert to numpy array (ensuring proper sample width and sample rate)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (1 << (8 * audio.sample_width - 1))  # Normalize to [-1, 1]
        
        # Create meter and measure
        meter = pyln.Meter(audio.frame_rate)  # Use actual frame rate
        loudness = meter.integrated_loudness(samples)
        
        return loudness if not np.isnan(loudness) else -23.0
        
    except Exception as e:
        print(f"Warning: LUFS calculation failed: {str(e)}")
        return -23.0  # Fallback value

def normalize_audio_lufs(audio, target_lufs=-20):
    """Normalize audio to target LUFS"""
    try:
        current_lufs = calculate_lufs(audio)
        
        # Avoid extreme adjustments
        if current_lufs < -50 or current_lufs > 0:
            print(f"Warning: Unusual LUFS value detected ({current_lufs}), using minimal adjustment")
            return audio
            
        gain_db = target_lufs - current_lufs
        
        # Limit maximum gain adjustment
        gain_db = max(min(gain_db, 20), -20)
        
        return audio + gain_db
        
    except Exception as e:
        print(f"Warning: LUFS normalization failed: {str(e)}")
        return audio

def analyze_reference_levels(dirs, target_lufs=-20):
    """Analyze first few segments to determine level adjustment factor"""
    try:
        print("\nAnalyzing reference audio levels...")
        audio_files = [f for f in os.listdir(dirs['audio']) if f.endswith('.wav')][:5]  # Take first 5 WAV files
        
        if not audio_files:
            print("Warning: No reference WAV files found, using default scaling")
            return 0
        
        lufs_values = []
        for audio_file in audio_files:
            try:
                audio_path = os.path.join(dirs['audio'], audio_file)
                print(f"Processing file: {audio_path}")
                
                # Verify file exists and is readable
                if not os.path.exists(audio_path):
                    print(f"Warning: File does not exist: {audio_path}")
                    continue
                    
                # Load audio and check if it's valid
                audio = AudioSegment.from_wav(audio_path)
                if len(audio) == 0:
                    print(f"Warning: Empty audio file: {audio_path}")
                    continue
                
                # Calculate LUFS
                lufs = calculate_lufs(audio)
                if math.isnan(lufs) or math.isinf(lufs):
                    print(f"Warning: Invalid LUFS value for {audio_file}")
                    continue
                    
                lufs_values.append(lufs)
                print(f"File {audio_file}: {lufs:.1f} LUFS")
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
        
        if not lufs_values:
            print("Warning: No valid LUFS values calculated, using default scaling")
            return 0
            
        avg_lufs = np.mean(lufs_values)
        adjustment = target_lufs - avg_lufs
        print(f"Average LUFS: {avg_lufs:.1f}")
        print(f"Required adjustment: {adjustment:+.1f} LUFS")
        return adjustment
        
    except Exception as e:
        print(f"Error in analyze_reference_levels: {str(e)}")
        print("Using default scaling")
        return 0

def clamp_audio_levels(audio, min_lufs=-26, max_lufs=-14):
    """Ensure audio stays within acceptable LUFS range"""
    try:
        current_lufs = calculate_lufs(audio)
        
        # Only adjust if the value seems reasonable
        if current_lufs < -50 or current_lufs > 0:
            return audio
            
        if current_lufs < min_lufs:
            gain_needed = min(min_lufs - current_lufs, 20)  # Limit maximum gain
            audio = audio + gain_needed
        elif current_lufs > max_lufs:
            gain_needed = max(max_lufs - current_lufs, -20)  # Limit maximum reduction
            audio = audio + gain_needed
        
        return audio
        
    except Exception as e:
        print(f"Warning: Level clamping failed: {str(e)}")
        return audio

def process_audio_segments(buffer_time=200, min_duration=1850, max_duration=8000):
    """Main processing function with sentence merging logic"""
    try:
        print("\nProcessing audio segments...")
        Config.initialize_directories()
        
        dirs = {
            'audio': Config.AUDIO_DIR,
            'srt': Config.SRT_DIR,
            'segmented_audio': Config.SEGMENTED_AUDIO_DIR,
            'training_data': Config.TRAINING_DATA_DIR
        }
        
        # Calculate reference levels
        lufs_adjustment = analyze_reference_levels(dirs)
        
        # Get sorted input files
        audio_files = sorted([f for f in os.listdir(dirs['audio']) if f.endswith('.wav')], 
                           key=lambda x: int(x.split('.')[0]))
        
        total_segments = 0
        total_duration = 0
        processed_segments = 0  # New counter for all processed segments
        dropped_segments = 0    # New counter for dropped segments
        
        # Add tracking of used filenames
        used_filenames = set()
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                audio_num = int(audio_file.split('.')[0])
                srt_file = os.path.join(dirs['srt'], f"{audio_num}.srt")
                
                print(f"\n\n=== Processing {audio_file} ===")
                print(f"SRT file: {srt_file}")
                
                if not os.path.exists(srt_file):
                    print(f"Warning: No SRT file found for {audio_file}")
                    continue
                
                # Load audio and apply LUFS adjustment
                audio = AudioSegment.from_wav(os.path.join(dirs['audio'], audio_file))
                
                # Load subtitles and print them
                subs = pysrt.open(srt_file)
                print("\nOriginal SRT contents:")
                for sub in subs:
                    print(f"{sub.index}: {sub.start} -> {sub.end}: {sub.text}")
                
                # Initialize variables for segment merging
                current_merged = None
                current_text = []
                current_start_ms = None
                segment_count = 0
                
                # Process each subtitle
                for i, sub in enumerate(subs):
                    start_ms = (sub.start.hours * 3600000 + 
                              sub.start.minutes * 60000 + 
                              sub.start.seconds * 1000 + 
                              sub.start.milliseconds)
                    end_ms = (sub.end.hours * 3600000 + 
                            sub.end.minutes * 60000 + 
                            sub.end.seconds * 1000 + 
                            sub.end.milliseconds)
                    
                    print(f"\nProcessing subtitle {i+1}:")
                    print(f"Time: {start_ms}ms -> {end_ms}ms")
                    print(f"Text: {sub.text.strip()}")
                    
                    segment = audio[start_ms:end_ms]
                    segment = clean_segment(segment)
                    
                    # If this is the first segment
                    if current_merged is None:
                        current_merged = segment
                        current_text = [sub.text.strip()]
                        current_start_ms = start_ms
                        print("Starting new merged segment")
                        continue
                    
                    # Check if adding this segment would exceed max_duration
                    potential_duration = end_ms - current_start_ms
                    
                    print(f"Current merged duration: {len(current_merged)}ms")
                    print(f"This segment duration: {len(segment)}ms")
                    print(f"Potential merged duration: {potential_duration}ms")
                    
                    if potential_duration <= max_duration:
                        # Merge segments by taking audio from start of first to end of current
                        current_merged = audio[current_start_ms:end_ms]
                        current_merged = clean_segment(current_merged)
                        current_text.append(sub.text.strip())
                        print("Merged with current segment")
                        print(f"Current text: {' '.join(current_text)}")
                    else:
                        # Process and save current merged segment
                        processed = add_silence_buffers(current_merged, buffer_time)
                        processed, orig_dur, final_dur = optimize_silence(processed)
                        
                        print("\nSaving merged segment:")
                        print(f"Original duration: {orig_dur}ms")
                        print(f"Final duration: {final_dur}ms")
                        print(f"Text: {' '.join(current_text)}")
                        
                        if min_duration <= len(processed) <= max_duration:
                            # Generate unique filename
                            base_filename = f"{audio_num:03d}_{segment_count + 1:03d}.wav"
                            counter = 1
                            while base_filename in used_filenames:
                                counter += 1
                                base_filename = f"{audio_num:03d}_{segment_count + 1:03d}_{counter}.wav"
                            
                            used_filenames.add(base_filename)
                            segment_path = os.path.join(dirs['segmented_audio'], base_filename)
                            
                            # Log the segment being saved
                            print(f"Saving segment: {base_filename}")
                            processed.export(segment_path, format='wav')
                            
                            with open(os.path.join(dirs['training_data'], 'output.txt'), 'a', encoding='utf-8') as f:
                                f.write(f"{base_filename}|{' '.join(current_text)}|1\n")
                            
                            segment_count += 1
                            total_segments += 1
                            total_duration += len(processed)
                        else:
                            dropped_segments += 1  # Count dropped segments
                        processed_segments += 1    # Count all processed segments
                        
                        # Start new segment with current subtitle
                        current_merged = segment
                        current_text = [sub.text.strip()]
                        current_start_ms = start_ms
                
                # Process final segment if it exists
                if current_merged is not None:
                    processed = add_silence_buffers(current_merged, buffer_time)
                    processed, orig_dur, final_dur = optimize_silence(processed)
                    
                    if min_duration <= len(processed) <= max_duration:
                        # Generate unique filename
                        base_filename = f"{audio_num:03d}_{segment_count + 1:03d}.wav"
                        counter = 1
                        while base_filename in used_filenames:
                            counter += 1
                            base_filename = f"{audio_num:03d}_{segment_count + 1:03d}_{counter}.wav"
                        
                        used_filenames.add(base_filename)
                        segment_path = os.path.join(dirs['segmented_audio'], base_filename)
                        
                        # Log the segment being saved
                        print(f"Saving segment: {base_filename}")
                        processed.export(segment_path, format='wav')
                        
                        with open(os.path.join(dirs['training_data'], 'output.txt'), 'a', encoding='utf-8') as f:
                            f.write(f"{base_filename}|{' '.join(current_text)}|1\n")
                        
                        total_duration += len(processed)
                        total_segments += 1
                    else:
                        dropped_segments += 1  # Count dropped segments
                    processed_segments += 1    # Count all processed segments
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
        
        # Add final statistics
        success_rate = (total_segments / processed_segments * 100) if processed_segments > 0 else 0
        print(f"\nProcessing complete:")
        print(f"Processed {processed_segments} segments total")
        print(f"Successfully saved {total_segments} segments")
        print(f"Dropped {dropped_segments} segments (duration constraints)")
        print(f"Success rate: {success_rate:.1f}%")
        
        return total_segments, total_duration
        
    except Exception as e:
        print(f"Error in process_audio_segments: {str(e)}")
        raise

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