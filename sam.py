import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import math
# select the device for computation

if torch.cuda.is_available():
    device = torch.device("cuda")
# MPS seems to crash every now and then
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")



if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator_1 = SAM2AutomaticMaskGenerator(sam2)



def downscale_image_by_percentage(image, scale_percent):
    """
    Downscale the image by a percentage while maintaining the aspect ratio.

    Args:
        image (PIL.Image or numpy.ndarray): The input image to downscale.
        scale_percent (float): The percentage to scale the image by (e.g., 50 for 50% of the original size).

    Returns:
        PIL.Image: The downscaled image.
    """
    # If the image is in NumPy format, convert it back to a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Calculate the new size based on the scale percentage
    width, height = image.size
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    new_size = (new_width, new_height)

    # Resize the image to the new size
    downscaled_image = image.resize(new_size,  Image.Resampling.LANCZOS)

    return downscaled_image

def load_image(path):
    image = Image.open(path)
    image = image.convert("RGB")
    image = downscale_image_by_percentage(image, scale_percent=100)
    image = np.array(image.convert("RGB"))
    return image

image1 = load_image('./subset_images/IMG_20240820_074416.jpg')
image2 = load_image('./subset_images/IMG_20240616_104257.jpg')





mask_generator=mask_generator_2

masks1 = mask_generator.generate(image1)

def extract_segment_from_image(image, mask):
    """
    Extract the segment from the base image using the provided binary mask.

    Args:
        image (PIL.Image or numpy.ndarray): The input base image.
        mask (numpy.ndarray): The binary mask where True or 1 values represent the region to extract.

    Returns:
        PIL.Image: The extracted segment of the image.
    """
    # Ensure image is in NumPy array format
    if isinstance(image, Image.Image):
        image = np.array(image)

    # If the mask is in binary format (0, 1 or True/False), ensure it's boolean
    # mask = mask.astype(bool)

    # Create a blank image with the same size as the input image
    segment = np.zeros_like(image)

    # Copy the image where the mask is True (extract the segment)
    segment[mask] = image[mask]

    # Convert the segment back to PIL.Image format if necessary
    return Image.fromarray(segment)


def plot_segments_in_grid(image, filtered_masks):
    """
    Plots all the extracted segments from filtered masks in a grid layout.

    Args:
        image (PIL.Image or numpy.ndarray): The input base image.
        filtered_masks (list): A list of masks (output of SAM) containing 'segmentation' key.
        grid_size (tuple): The number of rows and columns for the grid layout.
    """
    # Create a matplotlib figure for plotting the grid
    grid_columns = 5
    grid_rows = math.ceil(len(filtered_masks) / grid_columns)
    fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(5 * grid_columns, 5 * grid_rows))

    # Flatten the axes for easy indexing
    axes = axes.flatten()

    # Loop through the filtered masks and extract each segment
    for idx, mask in enumerate(filtered_masks):
        if idx >= grid_columns * grid_rows:
            print("More masks than grid slots. Some masks may not be displayed.")
            break

        # Get the segmentation mask for the current mask
        segmentation = mask['segmentation']

        # Extract the segment from the base image using the mask
        extracted_segment = extract_segment_from_image(image, segmentation)

        # Plot the extracted segment in the grid
        axes[idx].imshow(extracted_segment)

        # bbox
        x_min, y_min, width, height = mask['bbox']
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='red', facecolor='none')
        axes[idx].add_patch(rect)

        axes[idx].axis('off')
        axes[idx].set_title(f"Mask {idx + 1}")

    # Turn off any unused axes
    for ax in axes[idx + 1:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

import supervision as sv
import matplotlib.patches as patches


surface1=image1.shape[0]*image1.shape[(1)]

def mask_info(image,masks):
    print(len(masks))
    print(masks[0].keys())
    surface=image.shape[0]*image.shape[(1)]
    print([(mask['area'] / surface, mask['bbox']) for mask in sorted(masks, key=lambda x: x['area']/surface, reverse=True)])

def plot_mask(masks):
    masks_segment = [ mask['segmentation'] for mask in sorted(masks, key=lambda x: x['area'], reverse=True)]
    sv.plot_images_grid(images=masks_segment[:36], grid_size=(6, 6), size=(12, 12) )

def mask_area_filter(image,masks,min_surf=0.01, max_surf=0.10):
    surface=image.shape[0]*image.shape[(1)]
    return [m for m in masks if ((m['area'] / surface) > min_surf and (m['area'] / surface) < max_surf)]

def plot_filtered_masks(image,masks):
    mask_info(image,masks)
    filtered_masks = mask_area_filter(image,masks,0.01,0.10)
    mask_info(image,filtered_masks)
    plot_segments_in_grid(image, filtered_masks)


plot_filtered_masks(image1,masks1)
plot_segments_in_grid(image1, masks1)


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def extract_segment_from_image(image, mask):
    """
    Extract the segment from the base image using the provided binary mask.
    The segment is cropped to its bounding box for efficient template matching.

    Args:
        image (numpy.ndarray): The input base image.
        mask (numpy.ndarray): The binary mask where True or 1 values represent the region to extract.

    Returns:
        numpy.ndarray: The cropped segment of the image.
    """
    # Ensure the mask is a boolean array
    mask = mask.astype(bool)

    # Find the bounding box of the mask (non-zero area)
    y_coords, x_coords = np.where(mask)
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("No mask found in the given image region.")

    # Get the bounding box of the mask
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Crop the segment to this bounding box
    cropped_segment = image[y_min:y_max+1, x_min:x_max+1]

    return cropped_segment


def find_matching_segment(template, target_image):
    """
    Use cv2.matchTemplate to find the location of the segment in the target image.

    Args:
        template (numpy.ndarray): The extracted segment (template) from the first image.
        target_image (numpy.ndarray): The target image in which to search for the template.

    Returns:
        tuple: Top-left corner of the best matching region in the target image, match score.
    """
    # Convert both the template and target image to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)

    # Perform template matching using cv2.matchTemplate
    result = cv2.matchTemplate(target_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location with the highest match score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # max_loc is the top-left corner of the best match
    return max_loc, max_val


import numpy as np
import cv2

def find_matching_segment_with_distance_penalty(template, target_image, template_bbox, penalty_factor=0.001):
    """
    Use cv2.matchTemplate to find the location of the segment in the target image,
    and penalize the match score based on how far the match is from the original template's bounding box.

    Args:
        template (numpy.ndarray): The extracted segment (template) from the first image.
        target_image (numpy.ndarray): The target image in which to search for the template.
        template_bbox (tuple): The bounding box of the template in the format (x_min, y_min, width, height).
        penalty_factor (float): A factor to control how much distance affects the score.

    Returns:
        tuple: Top-left corner of the best matching region in the target image, penalized match score.
    """
    # Convert both the template and target image to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)

    # Perform template matching using cv2.matchTemplate
    result = cv2.matchTemplate(target_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location with the highest match score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # max_loc is the top-left corner of the best match
    matched_top_left = max_loc

    # Extract the top-left corner of the original template's bounding box
    template_top_left = (template_bbox[0], template_bbox[1])

    # Calculate the Euclidean distance between the matched location and the template's original location
    distance = np.linalg.norm(np.array(matched_top_left) - np.array(template_top_left))

    # Apply a penalty to the match score based on the distance
    penalized_score = max_val - (penalty_factor * distance)

    # Ensure the penalized score doesn't go below a certain threshold (e.g., 0)
    penalized_score = max(0, penalized_score)

    return matched_top_left, penalized_score



def process_all_masks(image, masks, target_image):
    """
    Process all masks, extract segments from the base image, and find corresponding matching regions in the target image.

    Args:
        image (numpy.ndarray): The input base image.
        masks (list): List of SAM-generated mask results (each containing 'segmentation').
        target_image (numpy.ndarray): The target image to search for matching regions.

    Returns:
        list: List of dictionaries with information about each match.
    """
    results = []

    # Loop over all the masks
    for idx, mask_data in enumerate(masks):
        # Extract the segmentation mask from the mask_data
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        # Extract the segment from the base image using the mask
        extracted_segment = extract_segment_from_image(image, mask)

        # Find the corresponding matching region in the target image
        # best_match_loc, match_score = find_matching_segment(extracted_segment, target_image)
        best_match_loc, match_score = find_matching_segment_with_distance_penalty(extracted_segment, target_image, bbox)

        # Store the result with necessary information
        results.append({
            'mask_index': idx,
            'best_match_loc': best_match_loc,
            'match_score': match_score,
            'segment_shape': extracted_segment.shape[:2]  # Height, width of the segment
        })

    return results

import matplotlib.patches as patches

def plot_matches_side_by_side(base_image, target_image, match_results, masks):
    """
    Plot the original base image with segments on the left and the matched segments on the target image on the right.

    Args:
        base_image (numpy.ndarray): The original base image from which segments were extracted.
        target_image (numpy.ndarray): The target image where matches were found.
        match_results (list): List of dictionaries containing match information for each mask.
        masks (list): List of SAM-generated masks (with 'segmentation' and 'bbox').
    """
    # Create a copy of both images for displaying
    base_image_copy = base_image.copy()
    target_image_copy = target_image.copy()

    # Create a matplotlib figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the base image with segment outlines on the left
    axes[0].imshow(base_image_copy)
    axes[0].set_title("Original Image with Segments")

    # Plot the target image with match rectangles on the right
    axes[1].imshow(target_image_copy)
    axes[1].set_title("Matched Segments on Target Image")

    # Loop through the match results and draw bounding boxes for both images
    for idx, (result, mask_data) in enumerate(zip(match_results, masks)):
        # Extract the original mask and bounding box (for the base image)
        mask_bbox = mask_data['bbox']
        x_min, y_min, width, height = mask_bbox

        # Draw rectangle and index in the center of the segment in the base image
        rect_base = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='blue', facecolor='none')
        axes[0].add_patch(rect_base)

        # Calculate center of the bounding box
        center_x_base = x_min + width / 2
        center_y_base = y_min + height / 2

        # Add index in the center of the base image's bounding box
        axes[0].text(center_x_base, center_y_base, str(idx), color='white', fontsize=12, ha='center', va='center')

        # Draw rectangles around the best match location on the target image
        top_left = result['best_match_loc']
        h, w = result['segment_shape']  # Height and width of the segment
        rect_target = patches.Rectangle(top_left, w, h, linewidth=2, edgecolor='green', facecolor='none')
        axes[1].add_patch(rect_target)

        # Calculate center of the bounding box on the target image
        center_x_target = top_left[0] + w / 2
        center_y_target = top_left[1] + h / 2

        # Add index in the center of the target image's bounding box
        axes[1].text(center_x_target, center_y_target, str(idx), color='white', fontsize=12, ha='center', va='center')

    # Hide axis ticks for both subplots
    axes[0].axis('off')
    axes[1].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def match_and_plot_segments(image,masks,target_image,min_segment_area=0.003,max_segment_area=0.10, keep_results=5):
    filtered_masks = mask_area_filter(image, masks, min_segment_area, max_segment_area)
    results = process_all_masks(image, filtered_masks, target_image)
    top_results = sorted(results, key=lambda x: x['match_score'], reverse=True)[:keep_results]
    top_masks = [filtered_masks[result['mask_index']] for result in top_results]
    plot_matches_side_by_side(image, target_image, top_results, top_masks)
    return top_results,top_masks,filtered_masks


results2, top_masks2, masks2 = match_and_plot_segments(image1, masks1, image2)



import cv2
import numpy as np

def calculate_bounding_box_center(bbox):
    """
    Calculate the center of a bounding box.

    Args:
        bbox (tuple): Bounding box in the format (x_min, y_min, width, height).

    Returns:
        tuple: Center point (x, y) of the bounding box.
    """
    x_min, y_min, width, height = bbox
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    return center_x, center_y

def calculate_homography(template_bboxes, target_bboxes):
    """
    Calculate the homography matrix to warp the target image to match the template.

    Args:
        template_bboxes (list of tuples): List of bounding boxes in the template image (x_min, y_min, width, height).
        target_bboxes (list of tuples): List of bounding boxes in the target image (x_min, y_min, width, height).

    Returns:
        numpy.ndarray: The 3x3 homography matrix.
    """
    # Calculate the centers of the bounding boxes
    template_points = np.array([calculate_bounding_box_center(bbox) for bbox in template_bboxes])
    target_points = np.array([calculate_bounding_box_center(bbox) for bbox in target_bboxes])

    # Find the homography matrix using the points
    H, status = cv2.findHomography(target_points, template_points, cv2.RANSAC)

    return H

def warp_target_image(target_image, homography_matrix, template_image_size):
    """
    Warp the target image using the homography matrix.

    Args:
        target_image (numpy.ndarray): The target image to be warped.
        homography_matrix (numpy.ndarray): The 3x3 homography matrix.
        template_image_size (tuple): The size of the template image (width, height).

    Returns:
        numpy.ndarray: The warped target image.
    """
    # Warp the target image to align with the template
    warped_image = cv2.warpPerspective(target_image, homography_matrix, template_image_size)

    return warped_image

# Example usage
def align_images_using_homography(base_image, target_image, match_results, masks):
    """
    Align the target image to the base image using homography based on matched bounding boxes.

    Args:
        base_image (numpy.ndarray): The base image (template).
        target_image (numpy.ndarray): The target image to be warped.
        match_results (list): List of dictionaries containing match information for each mask.
        masks (list): List of SAM-generated masks (with 'bbox' key).

    Returns:
        numpy.ndarray: The warped target image.
    """
    # Extract the bounding boxes from the masks and match results
    template_bboxes = [masks[result['mask_index']]['bbox'] for result in match_results]
    target_bboxes = [(result['best_match_loc'][0], result['best_match_loc'][1], result['segment_shape'][1], result['segment_shape'][0]) for result in match_results]
    print(template_bboxes)
    print(target_bboxes)
    # Calculate the homography matrix
    H = calculate_homography(template_bboxes, target_bboxes)

    # Get the size of the base (template) image
    template_image_size = (base_image.shape[1], base_image.shape[0])  # (width, height)

    # Warp the target image to align with the base image
    warped_image = warp_target_image(target_image, H, template_image_size)

    return warped_image


import matplotlib.pyplot as plt
cv2.imwrite('/tmp/image1.jpg', image1)

warped_target_image = align_images_using_homography(image1, image2, results2, masks2)
cv2.imwrite('/tmp/image2.jpg', warped_target_image)
