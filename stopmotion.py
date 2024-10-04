import cv2
import numpy as np
import os
import subprocess

def create_stop_motion_movie(input_folder, output_file, frame_duration=2, transition_duration=1, fps=30):
    # Get all jpg files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    image_files.sort()  # Sort the files to ensure correct order

    # Get the dimensions of the first image
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width = first_image.shape[:2]

    # Create a temporary video file
    temp_output = 'temp_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    for i in range(len(image_files)):
        current_img = cv2.imread(os.path.join(input_folder, image_files[i]))
        next_img = cv2.imread(os.path.join(input_folder, image_files[(i + 1) % len(image_files)]))
        
        # Hold the current image
        for _ in range(fps * frame_duration):
            out.write(current_img)
        
        # Cross-fade to the next image
        for j in range(int(fps * transition_duration)):
            alpha = j / (fps * transition_duration)
            blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
            out.write(blended)

    out.release()

    # Use FFmpeg to convert the temporary video to the final output with improved compression
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', temp_output,
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '23',
        '-vf', f'scale=-2:720',  # Scale to 720p, maintaining aspect ratio
        '-movflags', '+faststart',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_file
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # Remove the temporary file
    os.remove(temp_output)

    print(f"Stop motion movie created: {output_file}")

# Usage example
input_folder = 'warped'
output_file = 'parkbridge.mp4'
create_stop_motion_movie(input_folder, output_file, transition_duration=0.75)