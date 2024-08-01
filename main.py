import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Parameters
area_size = (100, 100)  # Define a 100x100 grid
depth_range = (0, 50)   # Depths between 0 and 50 meters
num_samples = 1000      # Number of samples

# Generate grid coordinates
x = np.linspace(0, area_size[0], area_size[0])
y = np.linspace(0, area_size[1], area_size[1])
x, y = np.meshgrid(x, y)

# Simulate depth data with random values
np.random.seed(42)
depth = np.random.uniform(depth_range[0], depth_range[1], area_size)

# Create a synthetic multispectral image with noise
def generate_multispectral_image(depth, noise_level=0.1):
    channels = 3  # Simulate 3 spectral channels (e.g., RGB)
    image = np.stack([depth + np.random.normal(0, noise_level, depth.shape) for _ in range(channels)], axis=-1)
    return image

# Simulate multispectral images
multispectral_image = generate_multispectral_image(depth)

# Flatten the data for machine learning
flat_x = x.flatten()
flat_y = y.flatten()
flat_depth = depth.flatten()
flat_image = multispectral_image.reshape(-1, 3)

# Create a DataFrame
data = pd.DataFrame({
    'x': flat_x,
    'y': flat_y,
    'depth': flat_depth,
    'channel_1': flat_image[:, 0],
    'channel_2': flat_image[:, 1],
    'channel_3': flat_image[:, 2]
})

# Sample the data to create a training set
sample_indices = np.random.choice(data.index, num_samples, replace=False)
sampled_data = data.loc[sample_indices]

# Normalize the spectral data for better training
scaler = MinMaxScaler()
sampled_data[['channel_1', 'channel_2', 'channel_3']] = scaler.fit_transform(sampled_data[['channel_1', 'channel_2', 'channel_3']])

# Show the first few rows of the simulated data
print(sampled_data.head())

# Visualize the synthetic depth map and a channel of the multispectral image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Simulated Depth Map")
plt.imshow(depth, cmap='viridis')
plt.colorbar(label='Depth (m)')

plt.subplot(1, 2, 2)
plt.title("Simulated Spectral Channel 1")
plt.imshow(multispectral_image[:, :, 0], cmap='gray')
plt.colorbar(label='Intensity')

plt.show()
