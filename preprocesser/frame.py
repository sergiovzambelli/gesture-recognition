import cv2
import glob
import os

def extract_frames(video_path, output_root_dir, target_fps=30):
    """
    Extracts frames from a video at a specified target FPS.
    Creates a subdirectory for each video (named after the video file)
    within output_root_dir, and names frames sequentially (1, 2, 3...).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval for extraction
    if original_fps == 0:
        frame_interval = 1
    else:
        frame_interval = max(1, round(original_fps / target_fps)) # Ensure at least 1 frame is extracted


    # Create output directory for this video's frames
    # This remains the same as it correctly creates a subfolder named after the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    frame_count_in_video = 0 # Counter for frames read from the original video
    extracted_sequence_number = 1 # Counter for extracted frames, starting from 1 for each new video

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        if frame_count_in_video % frame_interval == 0:
            # Changed filename format to sequential numbers starting from 1
            frame_filename = os.path.join(output_dir, f"{extracted_sequence_number}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_sequence_number += 1 # Increment for the next extracted frame

        frame_count_in_video += 1 # Always increment for each frame processed from video

    cap.release()


if __name__ == "__main__":
    # --- Configuration ---
    # The folder containing your clipped videos (e.g., from the previous FFmpeg step)
    source_video_folder = "clipped/no_wave"

    # The root directory where extracted frames will be saved.
    # For each video in source_video_folder, a subfolder named after the video
    # will be created here (e.g., 'frames/wave/video1_part_001/', 'frames/wave/video1_part_002/', etc.)
    output_base_directory = "frames/no_wave"

    # Desired FPS for the extracted frames
    desired_extraction_fps = 30
    # --- End Configuration ---

    # List to store all video file paths found
    video_files_to_process = []
    # Common video file extensions, including .MP4 for case sensitivity
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.flv', '*.webm', '*.MP4']

    # Use glob to find all video files within the specified source folder
    print(f"Scanning for videos in: {source_video_folder}")
    for ext in video_extensions:
        video_files_to_process.extend(glob.glob(os.path.join(source_video_folder, ext)))

    if not video_files_to_process:
        print(f"No video files found in '{source_video_folder}' with extensions: {', '.join(video_extensions)}")
    else:
        # Create the base output directory if it doesn't exist
        os.makedirs(output_base_directory, exist_ok=True)

        # Process each found video file
        for video_file in video_files_to_process:
            extract_frames(video_file, output_base_directory, desired_extraction_fps)
