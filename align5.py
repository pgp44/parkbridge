import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import piexif
from datetime import datetime

# Global variables for mouse interaction
selected_regions = []
start_point = None
dragging = False

def crop_center(image, crop_width, crop_height):
    """Crops the image from the center with given width and height."""
    img_width, img_height = image.size
    left = (img_width - crop_width) // 2
    top = (img_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))


def select_regions(event, x, y, flags, param):
    """Callback function to capture mouse drag and select rectangular regions."""
    global start_point, dragging, selected_regions
    reference_image, reference_image_copy = param  # Get both the reference image and its copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the rectangle
        start_point = (x, y)
        dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Redraw the image and draw the rectangle during dragging
        reference_image_copy[:] = reference_image.copy()  # Reset to the original image before drawing
        cv2.rectangle(reference_image_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select Stable Regions", reference_image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing the rectangle
        end_point = (x, y)
        dragging = False
        selected_regions.append((start_point, end_point))
        cv2.rectangle(reference_image_copy, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Select Stable Regions", reference_image_copy)

def select_stable_regions(reference_image):
    """Allow user to manually select stable regions in the reference image by dragging the mouse."""
    global selected_regions
    selected_regions.clear()  # Clear previous selections
    reference_image_copy = reference_image.copy()

    # Display the image and set up the callback
    cv2.imshow("Select Stable Regions", reference_image_copy)
    cv2.setMouseCallback("Select Stable Regions", select_regions, param=(reference_image, reference_image_copy))

    cv2.waitKey(0)  # Wait until the user presses any key
    cv2.destroyAllWindows()

    return selected_regions

def find_corresponding_regions(image, reference_regions, reference_image):
    """Automatically find corresponding regions in the given image using template matching."""
    corresponding_regions = []
    img_height, img_width = image.shape[:2]

    for (x1, y1), (x2, y2) in reference_regions:
        # Extract the template from the reference image
        template = reference_image[y1:y2, x1:x2]
        template_height, template_width = template.shape[:2]

        # Ensure the region in the image matches the template size by resizing the image
        if template_height > img_height or template_width > img_width:
            print(f"Resizing image region to match the template size: {template_width}x{template_height}.")
            resized_image = cv2.resize(image, (template_width, template_height), interpolation=cv2.INTER_AREA)
        else:
            resized_image = image

        # Apply template matching on the resized image
        res = cv2.matchTemplate(resized_image, template, cv2.TM_CCOEFF_NORMED)

        # Find the best match location
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

        # Append the region as a tuple of two points
        corresponding_regions.append((top_left, bottom_right))

    return corresponding_regions


def opencv_to_pil(opencv_image):
    """Convert OpenCV image (BGR format) to PIL image (RGB format)."""
    # Convert BGR (OpenCV) to RGB (Pillow)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def pil_to_opencv(pil_image):
    """Convert PIL image to OpenCV image (BGR format)."""
    # Convert the PIL image (which is RGB) to a NumPy array
    open_cv_image = np.array(pil_image)
    # Convert RGB (Pillow) to BGR (OpenCV)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image


def apply_homography_and_save(images_folder, output_folder, reference_image, regions_ref):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [img for img in os.listdir(images_folder) if img.endswith(('.jpg'))]
    for img_name in images:
        print(img_name)
        img_path = os.path.join(images_folder, img_name)
        pil_image = Image.open(img_path)

        if pil_image is None:
            print(f"Error: Could not load image {img_name}. Skipping.")
            continue
        
        image = pil_to_opencv(pil_image)

        # Automatically find corresponding regions in the current image using template matching
        regions_curr = find_corresponding_regions(image, regions_ref, reference_image)

        if len(regions_curr) != len(regions_ref):
            print(f"Skipping {img_name}, regions mismatch.")
            continue

        # Compute homography based on region centers (e.g., center of the rectangles)
        points_ref = np.float32([((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1), (x2, y2) in regions_ref])
        points_curr = np.float32([((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1), (x2, y2) in regions_curr])

        homography_matrix, _ = cv2.findHomography(points_curr, points_ref, cv2.RANSAC)

        # Warp the image using the homography matrix
        height, width = reference_image.shape[:2]
        warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))

        # Crop (applying the homography can result in black areas around the edges)
        warped_image = opencv_to_pil(warped_image)
        cropped_img = crop_center(warped_image, 3000, 1800)

        # Timestamp from exif
        exif_data = piexif.load(pil_image.info.get('exif', b''))
        exif_datetime = exif_data.get('Exif', {}).get(piexif.ExifIFD.DateTimeOriginal)
        if exif_datetime:
            dt = datetime.strptime(exif_datetime.decode('utf-8'), '%Y:%m:%d %H:%M:%S')
            timestamp_str = dt.strftime('%d-%b %H:%M')
            draw = ImageDraw.Draw(cropped_img)
            font_path = "/Library/Fonts/PTSans-Regular.ttf"
            font_size = 40
            font = ImageFont.truetype(font_path, font_size)
            img_width, img_height = cropped_img.size
            text_position = (img_width // 2, 80)  # Top center, 12 pixels from the top
            draw.text(text_position, timestamp_str, font=font, fill=(255, 255, 255), anchor="ms")
            
            # Convert back to OpenCV format
            warped_image_with_text = pil_to_opencv(cropped_img)
            cv2.imwrite(output_folder+'/'+img_name, warped_image_with_text)
        else:
            print('No exif datatime for ',img_name)
            cv2.imwrite(output_folder+'/'+img_name, cropped_img)


def main():
    # Set the folder paths
    images_folder = "images"
    output_folder = "images_warped"

    # Load the reference image
    reference_image_path = os.path.join(images_folder, "IMG_20240820_074416.jpg")
    reference_image = cv2.imread(reference_image_path)

    if reference_image is None:
        print(f"Error: Could not load reference image at {reference_image_path}")
        return

    # Step 1: Let the user select stable regions on the reference image
    print("Select stable regions on the reference image by dragging the mouse...")
    regions_ref = select_stable_regions(reference_image)

    if len(regions_ref) < 4:
        print("Error: You need to select at least 4 regions for homography.")
        return

    print("Processing images and applying transformations...")
    apply_homography_and_save(images_folder, output_folder, reference_image, regions_ref)
    
if __name__ == "__main__":
    main()
