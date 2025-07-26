import subprocess
import os
import math
import time
import glob

def get_video_duration(video_path):
    """
    Gets the duration of a video file using ffprobe (part of FFmpeg).
    Returns duration in seconds (float).
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except subprocess.CalledProcessError as e:
        print(f"Error getting duration for {video_path}: {e.stderr}")
        return None
    except ValueError:
        print(f"Could not parse duration for {video_path}. Output: '{result.stdout.strip()}'")
        return None
    except FileNotFoundError:
        print("Error: ffprobe not found. Please ensure FFmpeg is installed and in your system's PATH.")
        return None

def split_video_into_chunks(video_path, output_dir, segment_duration_secs, current_clip_number):
    """
    Splits a video into segments of a specified duration.
    Segments shorter than segment_duration_secs (e.g., the last partial segment) will be deleted.
    Output files are named sequentially starting from current_clip_number.
    """
    if not os.path.exists(video_path):
        return current_clip_number # Return current number if video not found

    video_duration = get_video_duration(video_path)
    if video_duration is None:
        return current_clip_number # Return current number if duration cannot be determined


    num_segments = math.ceil(video_duration / segment_duration_secs)

    for i in range(num_segments):
        start_time = i * segment_duration_secs
        
        # New naming convention: sequential number .mp4
        output_filename = os.path.join(output_dir, f"{current_clip_number}.mp4")

        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(segment_duration_secs),
            '-c', 'copy',
            '-y',
            output_filename
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            time.sleep(0.1) # Small delay to ensure file system has written the file
            clipped_duration = get_video_duration(output_filename)

            if clipped_duration is None:
                current_clip_number += 1 # Increment even if duration check failed, to avoid name conflicts
            elif clipped_duration < segment_duration_secs:
                # Add a small tolerance for floating point comparisons
                if abs(clipped_duration - segment_duration_secs) > 0.1: # e.g., if it's clearly less than 2s
                    os.remove(output_filename)
                else:
                    # If it's effectively the target duration (within tolerance), keep it
                    current_clip_number += 1 # Increment only if kept
            else:
                # If it's long enough, keep it
                current_clip_number += 1 # Increment only if kept

        except subprocess.CalledProcessError as e:
            print(f"  - Error splitting {os.path.basename(video_path)} segment {i+1}: {e.stderr}")
            # Do not increment current_clip_number if FFmpeg command failed
        except FileNotFoundError:
            print("Error:  or ffprobe not found. Please ensure FFmpeg is installed and in your system's PATH.")
            return current_clip_number # Return current number if FFmpeg not found

    return current_clip_number # Return the updated clip number


if __name__ == "__main__":
    target_folder_path = "no_wave"

    input_video_paths = []
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.flv', '*.webm', '*.MP4']
    
    for ext in video_extensions:
        # glob.glob returns a list of paths matching the pattern
        # os.path.join handles differences in path separators (e.g., / or \)
        input_video_paths.extend(glob.glob(os.path.join(target_folder_path, ext)))

    # All clipped videos will be saved directly into this directory
    output_base_directory = "clipped/no_wave/"

    segment_length_seconds = 2 # Each output clip will aim for this duration

    # Initialize the global counter for clip filenames
    global_clip_counter = 1
    # --- End Configuration ---

    # Create the base output directory if it doesn't exist
    os.makedirs(output_base_directory, exist_ok=True)

    for video_file_path in input_video_paths:
        # Pass the current counter and update it with the returned value
        global_clip_counter = split_video_into_chunks(
            video_file_path,
            output_base_directory,
            segment_length_seconds,
            global_clip_counter
        )
