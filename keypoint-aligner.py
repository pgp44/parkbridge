import cv2
import numpy as np
import os
import argparse
import piexif
from piexif import ExifIFD
from PIL import Image, ImageDraw, ImageFont
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
from datetime import datetime
import locale
import re

def add_timestamp_from_exif(image_path, image):
    if(os.path.exists(image_path)):
        orig_image = Image.open(image_path)
        exif_data = piexif.load(orig_image.info.get('exif', b''))
        exif_datetime = exif_data.get('Exif', {}).get(piexif.ExifIFD.DateTimeOriginal)
    
    if os.path.exists(image_path) and exif_datetime:
        dt = datetime.strptime(exif_datetime.decode('utf-8'), '%Y:%m:%d %H:%M:%S')
        timestamp_str = dt.strftime('%d-%b %H:%M')

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Add timestamp using PIL
        draw = ImageDraw.Draw(pil_image)
        font_path = "/Library/Fonts/PTSans-Regular.ttf"
        font_size = 40
        font = ImageFont.truetype(font_path, font_size)
        img_width, img_height = pil_image.size
        text_position = (img_width // 2, 80)
        draw.text(text_position, timestamp_str, font=font, fill=(255, 255, 255), anchor="ms")

        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result_image
    else:
        print('No EXIF datetime for the image')
        return image


def extract_start_date(cropped_image_folder):
    files = sorted(
        [f for f in os.listdir(cropped_image_folder) if f.startswith("IMG_") and f.endswith(".jpg")]
    )
    if not files:
        raise ValueError("No files found in cropped image folder.")
    first_file = files[0]
    date_str = first_file.split('_')[1]  # Extract 'YYYYMMDD' part
    start_date = datetime.strptime(date_str, "%Y%m%d")
    return start_date

def crop_center(image, crop_width, crop_height):
    img_width, img_height = image.size
    left = (img_width - crop_width) // 2
    top = (img_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

def save_crop_exif(src_image_path,crop_width, crop_height, cropped_images_folder):
    img = cv2.imread(src_image_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)    
    cropped_img = crop_center(pil_image, crop_width, crop_height)
    opencv_cropped_img = np.array(cropped_img)
    opencv_cropped_img_bgr = opencv_cropped_img
    src_image_path = re.sub(r"_reference", "", src_image_path)
    final_img = add_timestamp_from_exif(f"{src_image_path}", opencv_cropped_img_bgr)
    final_img = Image.fromarray(np.array(final_img))
    final_img.save(f'{cropped_images_folder}/{os.path.basename(src_image_path)}')

def crop_and_add_title(images_folder, crop_width, crop_height, cropped_with_title_images_folder):
    for file_name in sorted(os.listdir(images_folder)) :
        if file_name.lower().endswith(('.jpg')):
            image_file_path = os.path.join(images_folder, file_name)
            save_crop_exif(image_file_path, crop_width,crop_height,cropped_with_title_images_folder)

def convert_pedometer_file(input_file, start_date):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Skip the first two lines and process the rest
    data = []
    cumulative_steps = 0

    for line in lines[2:]:  # Skip the first two lines
        parts = line.strip().split(',')
        date = datetime.strptime(parts[0], "%Y%m%d")
        
        # Skip dates earlier than the start_date
        if date < start_date:
            continue
        
        daysteps = sum(map(int, parts[1:25]))
        cumulative_steps += daysteps
        data.append([date, daysteps, cumulative_steps])

    df = pd.DataFrame(data, columns=["date", "daysteps", "steps"])
    return df


def create_stop_motion_movie_with_steps(input_folder, output_file, steps, frame_duration=2, transition_duration=1, fps=30, audio_file=None):
    # Get all jpg files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    image_files.sort()
    # print(f"files {image_files}")

    # Get the dimensions of the first image
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width = first_image.shape[:2]

    temp_output = 'temp_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Initialize the line chart image
    line_chart_file = 'line_chart.png'
    full_data = steps.copy()

    # Loop through each image
    for i in range(len(image_files)):
        current_img = cv2.imread(os.path.join(input_folder, image_files[i]))
        next_img = cv2.imread(os.path.join(input_folder, image_files[(i + 1) % len(image_files)]))
        
        # Resize images to ensure they have the same size
        current_img = cv2.resize(current_img, (width, height))
        next_img = cv2.resize(next_img, (width, height))

        # Extract the date from the filename (format: IMG_yyyymmdd_hhmmss.jpg)
        image_date_str = image_files[i][4:12]  # Extract yyyymmdd
        image_date = datetime.strptime(image_date_str, '%Y%m%d')

        # Filter the CSV data up to the current image date
        filtered_data = steps[steps['date'] <= image_date]

        # Update the line chart with the filtered data (this keeps the line chart updated with new data)
        create_line_chart(filtered_data, full_data, line_chart_file)

        # Superimpose the line chart on the current image (without blending it)
        current_img_with_chart = overlay_line_chart(current_img, line_chart_file)

        # Hold the current image (with the updated chart)
        for _ in range(int(fps * frame_duration)):
            out.write(current_img_with_chart)

        # Cross-fade the images, but without fading the line chart
        for j in range(int(fps * transition_duration)):
            alpha = j / (fps * transition_duration)
            blended_img = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
            # Overlay the line chart on the blended image (to keep it static during the transition)
            blended_img_with_chart = overlay_line_chart(blended_img, line_chart_file)
            out.write(blended_img_with_chart)

    # Hold the last image if audio is longer than the video
    last_img = current_img_with_chart  # The last image with the chart
    video_duration = len(image_files) * (frame_duration + transition_duration)  # In seconds

    # Get the duration of the MP3 audio file
    if audio_file:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        audio_duration = float(result.stdout)
        
        if audio_duration > video_duration:
            hold_duration = audio_duration - video_duration
            print(f"Holding the last frame for {hold_duration} seconds.")
            # Hold the last image for the remaining duration of the audio
            for _ in range(int(hold_duration * fps)):
                out.write(last_img)
    out.release()

    # FFmpeg command to add the MP3 audio to the video
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', temp_output,            # Input: stop motion movie
        '-i', audio_file,             # Input: MP3 file
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '23',
        '-vf', f'scale=-2:720',  # Scale to 720p, maintaining aspect ratio
        '-movflags', '+faststart',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ar', '44100',            # Resample audio to 44.1 kHz
        '-shortest',               # Stops the video when the shorter stream ends (video or audio)
        output_file
    ]

    # Run the FFmpeg command
    subprocess.run(ffmpeg_cmd, check=True)

    # Remove the temporary file
    os.remove(temp_output)

    print(f"Stop motion movie with audio created: {output_file}")

def overlay_line_chart(image, line_chart_file):
    """
    Overlay the line chart in the bottom-right corner of the given image,
    resizing it to 750px wide and 450px high.
    """
    # Load the line chart as an image (with alpha channel for transparency)
    line_chart_img = cv2.imread(line_chart_file, cv2.IMREAD_UNCHANGED)  # Load with transparency

    # Resize the line chart to be 750px wide and 450px high
    line_chart_img_resized = cv2.resize(line_chart_img, (750, 450))

    # Get the dimensions of the resized chart
    chart_height, chart_width = line_chart_img_resized.shape[:2]

    # Define the region of interest (ROI) in the bottom-right corner
    x_offset = image.shape[1] - chart_width - 10  # 10 pixels from the right
    y_offset = image.shape[0] - chart_height - 10  # 10 pixels from the bottom

    # If the line chart has an alpha channel, blend it with the image
    if line_chart_img_resized.shape[2] == 4:
        # Split the line chart into its color channels and alpha channel
        b, g, r, alpha = cv2.split(line_chart_img_resized)

        # Normalize the alpha channel to be between 0 and 1
        alpha = alpha / 255.0

        # Blend the line chart with the image
        for c in range(0, 3):  # Iterate over the B, G, R channels
            image[y_offset:y_offset+chart_height, x_offset:x_offset+chart_width, c] = (
                alpha * line_chart_img_resized[:, :, c] +
                (1 - alpha) * image[y_offset:y_offset+chart_height, x_offset:x_offset+chart_width, c]
            )

    return image

def create_line_chart(filtered_data, full_data, output_file):
    """
    Creates a static line chart from filtered data (up to a specific date) and saves it as an image.
    The chart will have a light grey background with 70% opacity, display steps in 1K units with 1 decimal,
    and annotate the last data point with the date in Dutch format (dd-mon).
    """
    # Set locale to Dutch
    locale.setlocale(locale.LC_TIME, 'nl_NL.UTF-8')  # For Dutch date formatting

    fig, ax = plt.subplots(figsize=(4, 2))  # Create a smaller chart (400x200px)

    # Set light grey background for the figure with 70% opacity
    fig.patch.set_facecolor(mcolors.to_rgba('lightgrey', 0.7))
    ax.set_facecolor(mcolors.to_rgba('lightgrey', 0.7))

    # Plot the filtered data
    ax.plot(filtered_data['date'], filtered_data['steps'], lw=2)

    # Calculate 5% margin for x-axis extension
    date_range = full_data['date'].max() - full_data['date'].min()
    margin = date_range * 0.10  # 5% of the date range

    # Set the x-axis limits with the added margin
    ax.set_xlim(full_data['date'].min(), full_data['date'].max() + margin)

    # Set the y-axis limits
    ax.set_ylim(full_data['steps'].min() * 0.9, full_data['steps'].max() * 1.1)

    # Format y-axis to show steps in 1K units with 1 decimal
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.1f}K'))

    # Annotate the last data point if available, also in 1K units
    if not filtered_data.empty:
        last_date = filtered_data['date'].iloc[-1]
        last_value = filtered_data['steps'].iloc[-1] / 1000  # Convert to 1K units
        last_date_str = last_date.strftime('%d %b')  # Format date as 'dd-mon' in Dutch

        ax.annotate(f'{last_value:.1f}K\n{last_date_str}',  # Display value in 1K units with 1 decimal and date
                    xy=(last_date, last_value * 1000),  # Point at the last data point in original scale
                    xytext=(5, 5),  # Slightly offset the text
                    textcoords='offset points',
                    fontsize=10,
                    color='black')

    # Remove axis lines and labels
    ax.set_axis_off()

    # Save the chart as an image with a semi-transparent background
    plt.savefig(output_file, dpi=100)
    plt.close(fig)

def calculate_homography_metrics(H):
    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    shear = (H[0, 0] * H[1, 0] + H[0, 1] * H[1, 1]) / (scale_x * scale_y)
    determinant = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    u, s, vh = np.linalg.svd(H[:2, :2])  # Only consider the upper 2x2 for condition number
    condition_number = s[0] / s[-1] if s[-1] != 0 else np.inf
    perspective = np.linalg.norm(H[2, :2])
    rotation_angle = np.arctan2(H[1, 0], H[0, 0]) * (180 / np.pi)  # Convert to degrees
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear": shear,
        "determinant": determinant,
        "condition_number": condition_number,
        "perspective": perspective,
        "rotation_angle": rotation_angle
    }

def is_homography_nok(H):
    metrics = calculate_homography_metrics(H)
    return (metrics["condition_number"] > 10 or
            metrics["scale_x"] > 2 or metrics["scale_y"] > 2 or
            metrics["determinant"] < 0.1 or metrics["determinant"] > 2 or
            abs(metrics["rotation_angle"]) > 15)

def align_images(reference_image_path, image_folder, output_folder):
    not_aligned = 0
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f"Error: Unable to load reference image '{reference_image_path}'")
        return

    ref_gray_1 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.equalizeHist(ref_gray_1)

    sift = cv2.SIFT_create()

    ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_gray, None)

    bf = cv2.BFMatcher()

    for filename in os.listdir(image_folder):
        if filename == os.path.basename(reference_image_path):
            continue

        image_path = os.path.join(image_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            # print(f"'{output_path}' exists. Skipping alignment for '{filename}'...")
            continue

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Unable to load image '{image_path}', skipping.")
            continue

        img_gray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray_1)
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)
        matches = bf.knnMatch(ref_descriptors, descriptors, k=2)
        # Apply Lowe's ratio test to filter good matches
        threshold = 0.75
        good_matches = [m for m, n in matches if m.distance < threshold * n.distance]
        
        # Ensure there are enough good matches to compute homography
        if len(good_matches) > 4:
            ref_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            img_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0)
            if(is_homography_nok(H)):
                print(f"Homography out of bounds ({calculate_homography_metrics(H)}). Can't align {filename} with {reference_image_path} -> skipping")
                not_aligned = not_aligned + 1
                continue
            # Warp the current image to align with the reference image
            height, width = ref_img.shape[:2]
            aligned_img = cv2.warpPerspective(img, H, (width, height))

            # Convert aligned image to RGB format for saving with EXIF data
            aligned_img_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
            aligned_pil_img = Image.fromarray(aligned_img_rgb)

            # Extract EXIF data from the source image
            try:
                exif_dict = piexif.load(image_path)
                # Check and correct the SceneType tag if necessary
                if ExifIFD.SceneType in exif_dict['Exif']:
                    scene_type = exif_dict['Exif'][ExifIFD.SceneType]
                    if isinstance(scene_type, int):
                        exif_dict['Exif'][ExifIFD.SceneType] = bytes([scene_type])
                exif_bytes = piexif.dump(exif_dict)
            except Exception as e:
                print(f"Warning: Unable to extract EXIF data from '{image_path}': {e}")
                exif_bytes = None

            # Save the aligned image with EXIF data
            try:
                if exif_bytes:
                    aligned_pil_img.save(output_path, "JPEG", exif=exif_bytes)
                else:
                    aligned_pil_img.save(output_path, "JPEG")
                print(f"Aligned image saved as '{output_path}'")
            except Exception as e:
                print(f"Error: Unable to save aligned image '{output_path}': {e}")
                not_aligned = not_aligned + 1
                continue

            # # Uncomment the following lines to save match visualization images
            # match_img = cv2.drawMatches(ref_img, ref_keypoints, img, keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # match_output_path = os.path.join(output_folder, f"match_{filename}.png")
            # cv2.imwrite(match_output_path, match_img)
            # print(f"Match visualization saved as '{match_output_path}'")
        else:
            print(f"Warning: Not enough good matches found for '{filename} while attempting to align with {reference_image_path} -> skipping")
            not_aligned = not_aligned + 1
    return not_aligned

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Align images in a folder to a reference image.")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images to align.")
    parser.add_argument("--reference_image", required=True, help="Filename of the reference image within the folder.")
    parser.add_argument("--output_folder", help="Path to the folder where aligned images will be saved. Defaults to '<image_folder>_aligned'.")
    parser.add_argument("--step_counter", required=True, help="Path to file with 'pedometer' backup file")
    parser.add_argument("--crop_width", type=int, required=True, help="Width of the crop region (integer).")
    parser.add_argument("--crop_height", type=int, required=True, help="Height of the crop region (integer).")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file (string).")
    parser.add_argument("--stop_motion", type=str, required=True, help="Path to the stop-motion output file (string).")

    args = parser.parse_args()

    # Set default output folder if not provided
    if not args.output_folder:
        args.output_folder = f"{args.image_folder}_aligned"

    # Construct the full path to the reference image
    reference_image_path = os.path.join(args.image_folder, args.reference_image)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output directory '{args.output_folder}'")
    else:
        print(f"Output directory '{args.output_folder}' already exists")

    # Align images
    used_references = set()
    not_aligned = align_images(reference_image_path, args.image_folder, args.output_folder)
    used_references.add(reference_image_path)

    while not_aligned > 0:
        
        aligned_files = [
            os.path.normpath(os.path.join(args.output_folder, f))
            for f in os.listdir(args.output_folder)
            if os.path.normpath(os.path.join(args.output_folder, f)) not in {os.path.normpath(ref) for ref in used_references}
        ]
        
        print(f"# Used references: {len(used_references)}")
        print(f"# New reference candidates: {len(aligned_files)}")
        if not aligned_files:
            print(f"Giving up. {not_aligned} images left but no more references to try")
            break

        new_reference_image = aligned_files[0]
        print(f"{not_aligned} images left. Retrying with new reference ({new_reference_image})")
        not_aligned = align_images(new_reference_image, args.image_folder, args.output_folder)
        used_references.add(new_reference_image)

    print("Alignment process completed.")
    
    # Add crop and add date title
    cropped_image_folder = f"{args.output_folder}_cropped"
    os.makedirs(cropped_image_folder,exist_ok=True)
    crop_and_add_title(args.output_folder, args.crop_width, args.crop_height, f"{args.output_folder}_cropped")
    print("Cropping completed.")
    #
    first_image_date = extract_start_date(f"{args.output_folder}_cropped")
    steps=convert_pedometer_file(args.step_counter,first_image_date)
    #
    create_stop_motion_movie_with_steps(f"{args.output_folder}_cropped",args.stop_motion,steps,1.5,0.8,30,args.audio)
    print("Stop motion creation completed.")
