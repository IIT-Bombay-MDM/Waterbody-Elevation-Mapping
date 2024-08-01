import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

# Directory containing the VV and VH images
input_dir = 'data'

# Directory to save the dataset
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for images and masks
subdirs = ['images/train', 'images/val', 'images/test', 'masks/train', 'masks/val', 'masks/test']
for subdir in subdirs:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


# Function to get all VV and VH image file paths from multiple folders
def get_file_paths(root_dir, pattern):
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(pattern):
                file_paths.append(os.path.join(dirpath, filename))
    return sorted(file_paths)


# Get all VV and VH image file paths
vv_files = get_file_paths(input_dir, '_corrected_VV.tif')
vh_files = get_file_paths(input_dir, '_corrected_VH.tif')

# Ensure there are equal numbers of VV and VH images
assert len(vv_files) == len(vh_files), "The number of VV and VH images must be equal."

# Split the data into train, test, and validation sets
vv_train, vv_temp, vh_train, vh_temp = train_test_split(vv_files, vh_files, test_size=0.3, random_state=42)
vv_test, vv_val, vh_test, vh_val = train_test_split(vv_temp, vh_temp, test_size=0.5, random_state=42)

# Load the pre-trained EDSR model for super-resolution
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")
sr.setModel("edsr", 4)


# Function to apply super-resolution to increase sharpness
def apply_super_resolution(image):
    return sr.upsample(image)


def create_heatmap_mask(image, depth_thresholds, elevation_thresholds, depth_colors, elevation_colors):
    """
    Create a mask with different colors for different depth and elevation levels.

    Parameters:
    - image: Input image as a NumPy array.
    - depth_thresholds: List of thresholds for depth levels.
    - elevation_thresholds: List of thresholds for elevation levels.
    - depth_colors: List of RGB colors for depth levels.
    - elevation_colors: List of RGB colors for elevation levels.

    Returns:
    - mask: Colored mask as a NumPy array.
    """
    # Create an empty mask with 3 channels for RGB
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Apply depth colors
    for i in range(len(depth_thresholds) - 1):
        mask[(image >= depth_thresholds[i]) & (image < depth_thresholds[i + 1])] = depth_colors[i]

    # Apply elevation colors
    for i in range(len(elevation_thresholds) - 1):
        mask[(image >= elevation_thresholds[i]) & (image < elevation_thresholds[i + 1])] = elevation_colors[i]

    return mask


def add_legend_to_mask(mask, depth_thresholds, elevation_thresholds, depth_colors, elevation_colors):
    """
    Add legends for depth and elevation to the mask.

    Parameters:
    - mask: Mask image as a NumPy array.
    - depth_thresholds: List of thresholds for depth levels.
    - elevation_thresholds: List of thresholds for elevation levels.
    - depth_colors: List of RGB colors for depth levels.
    - elevation_colors: List of RGB colors for elevation levels.

    Returns:
    - mask_with_legend: Mask image with legends as a NumPy array.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask)

    # Create legends for depth
    for i, color in enumerate(depth_colors):
        ax.plot([], [], color=np.array(color) / 255, label=f'Depth: {depth_thresholds[i]}-{depth_thresholds[i + 1]}')

    # Create legends for elevation
    for i, color in enumerate(elevation_colors):
        ax.plot([], [], color=np.array(color) / 255,
                label=f'Elevation: {elevation_thresholds[i]}-{elevation_thresholds[i + 1]}')

    ax.legend(loc='upper right')
    plt.axis('off')
    fig.canvas.draw()

    mask_with_legend = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    mask_with_legend = mask_with_legend.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return mask_with_legend


# Function to process and save images
def process_and_save_images(vv_paths, vh_paths, subset_name):
    for vv_path, vh_path in zip(vv_paths, vh_paths):
        # Read the VV and VH images
        with rasterio.open(vv_path) as vv_src:
            vv_image = vv_src.read(1)

        with rasterio.open(vh_path) as vh_src:
            vh_image = vh_src.read(1)

        # Normalize the images
        vv_norm = (vv_image * 4 - vv_image.min())
        vh_norm = (vh_image * 4 - vh_image.min())

        # Create an RGB composite image
        rgb_image = np.zeros((vv_image.shape[0], vv_image.shape[1], 3), dtype=np.float32)
        rgb_image[:, :, 0] = vv_norm  # Red channel for VV
        rgb_image[:, :, 1] = (vv_norm + vh_norm)  # Green channel for VH
        rgb_image[:, :, 2] = (vv_norm + vh_norm) / 2  # Blue channel as a combination

        # Convert to 8-bit image for visualization
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Apply super-resolution to increase sharpness
        sr_rgb_image = apply_super_resolution(rgb_image)

        # Convert super-resolved image to grayscale
        sr_gray_image = cv2.cvtColor(sr_rgb_image, cv2.COLOR_BGR2GRAY)

        # Define depth and elevation thresholds and colors
        depth_thresholds = [0, 5, 20, 30, 35]
        elevation_thresholds = [36, 50, 65, 85, 100]
        depth_colors = [[0, 0, 255], [0, 255, 255], [255, 0, 0], [255, 0, 255]]
        elevation_colors = [[255, 255, 0], [255, 165, 0], [0, 128, 0], [0, 255, 255]]

        # Create heatmap mask
        mask = create_heatmap_mask(sr_gray_image, depth_thresholds, elevation_thresholds, depth_colors,
                                   elevation_colors)

        # Add legends to the mask
        mask_with_legend = add_legend_to_mask(mask, depth_thresholds, elevation_thresholds, depth_colors,
                                              elevation_colors)

        # Generate output filenames
        base_filename = os.path.basename(vv_path)
        trimmed_filename = '-'.join(
            base_filename.split('_')[4:])  # Adjust the slicing based on the actual filename pattern
        output_filename = trimmed_filename.replace('-corrected-VV.tif', '-composite.png')
        mask_filename = trimmed_filename.replace('-corrected-VV.tif', '-mask.png')

        # Save composite image
        output_image_path = os.path.join(output_dir, f'images/{subset_name}', output_filename)
        plt.imsave(output_image_path, sr_rgb_image)
        print(f"Saved composite image to {output_image_path}")

        # Save mask image
        output_mask_path = os.path.join(output_dir, f'masks/{subset_name}', mask_filename)
        mask_image = Image.fromarray(mask_with_legend)
        mask_image.save(output_mask_path)
        print(f"Saved mask image to {output_mask_path}")


# Process and save images for each subset
process_and_save_images(vv_train, vh_train, 'train')
process_and_save_images(vv_test, vh_test, 'test')
process_and_save_images(vv_val, vh_val, 'val')
