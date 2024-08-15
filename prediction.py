import os
import cv2
import numpy as np
from PIL import Image

# Directory where the generated super-resolved images are stored
dataset_dir = 'dataset/images/train'  # Change to 'val' or 'test' as needed

# Load the pre-trained EDSR model for super-resolution (if needed)
# sr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr.readModel("EDSR_x4.pb")
# sr.setModel("edsr", 4)

# Define the kernel size
kernel_size = 100

# Define the thresholds for suitable survival conditions
elevation_thresholds = (30, 85)
depth_thresholds = (0, 20)

# Define a function to predict human survival in a given 100x100 area
def predict_human_survival(area):
    # Convert the area to grayscale (if needed, based on the criteria)
    gray_area = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

    # Check for suitable elevation
    elevation_mask = (gray_area >= elevation_thresholds[0]) & (gray_area <= elevation_thresholds[1])

    # Check for suitable depth
    depth_mask = (gray_area >= depth_thresholds[0]) & (gray_area <= depth_thresholds[1])

    # Combine the masks
    combined_mask = elevation_mask & depth_mask

    # Calculate the percentage of suitable area
    suitable_percentage = np.sum(combined_mask) / (kernel_size * kernel_size)

    # Define a threshold for the minimum suitable percentage area
    suitable_threshold = 0.5  # 50% of the area should be suitable

    return suitable_percentage >= suitable_threshold

# Initialize an empty array to store predictions
image_paths = sorted([os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir) if img.endswith('.png')])

for image_path in image_paths:
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    predictions = np.zeros((height // kernel_size, width // kernel_size))

    # Slide the window across the image
    for i in range(0, height, kernel_size):
        for j in range(0, width, kernel_size):
            # Extract the current 100x100 area
            window = image[i:i+kernel_size, j:j+kernel_size]

            # Ensure the window is exactly 100x100
            if window.shape[0] != kernel_size or window.shape[1] != kernel_size:
                continue

            # Predict survival for the current window
            survival = predict_human_survival(window)

            # Store the prediction
            predictions[i // kernel_size, j // kernel_size] = survival

    # The predictions array now contains the survival prediction for each 100x100 area
    print(f"Predictions for {os.path.basename(image_path)}:")
    print(predictions)

    # Optionally save or visualize the predictions
    # np.savetxt(os.path.join('output_predictions', f"{os.path.basename(image_path)}.csv"), predictions, delimiter=",")
