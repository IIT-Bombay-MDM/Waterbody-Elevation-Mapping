import os
import numpy as np
import rasterio
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier

# Directory containing the VV and VH images
input_dir = 'data'
output_dir = 'output_dem'
os.makedirs(output_dir, exist_ok=True)


def get_file_paths(root_dir, pattern):
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(pattern):
                file_paths.append(os.path.join(dirpath, filename))
    return sorted(file_paths)


def phase_unwrapping(wrapped_phase):
    try:
        return unwrap_phase(wrapped_phase)
    except Exception as e:
        print(f"Error during phase unwrapping: {e}")
        raise


def calculate_ndvi(nir, red):
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi


def extract_features(vv_image, vh_image, ndvi):
    vv_vh_ratio = vv_image / (vh_image + 1e-10)
    vv_vh_diff = vv_image - vh_image
    features = np.stack([vv_image, vh_image, vv_vh_ratio, vv_vh_diff, ndvi], axis=-1)
    return features


def classify_terrain(features, model):
    flat_features = features.reshape(-1, features.shape[-1])
    classified_flat = model.predict(flat_features)
    classified = classified_flat.reshape(features.shape[:-1])
    return classified


def apply_colormap_and_save(dem, classified, output_path):
    try:
        cmap = plt.get_cmap('terrain')
        water_color = np.array([0, 0, 255])
        forest_color = np.array([34, 139, 34])
        settlement_color = np.array([139, 69, 19])

        rgb_image = np.zeros((*dem.shape, 3), dtype=np.uint8)
        rgb_image[classified == 1] = water_color
        rgb_image[classified == 2] = forest_color
        rgb_image[classified == 3] = settlement_color

        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=rgb_image.shape[0],
                width=rgb_image.shape[1],
                count=3,
                dtype=rgb_image.dtype,
                crs=dem_crs,
                transform=dem_transform,
        ) as dst:
            for i in range(3):
                dst.write(rgb_image[:, :, i], i + 1)

        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(rgb_image)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', edgecolor='blue', label='Water Bodies'),
            Patch(facecolor='green', edgecolor='green', label='Forest Cover'),
            Patch(facecolor='brown', edgecolor='brown', label='Human Settlements'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.axis('off')

        plt.savefig(output_path.replace('.tif', '_classified.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error during colormap application or saving: {e}")
        raise


def generate_dem(vv_path, vh_path, output_path, model):
    try:
        with rasterio.open(vv_path) as vv_src:
            vv_image = vv_src.read(1)

        with rasterio.open(vh_path) as vh_src:
            vh_image = vh_src.read(1)

        global dem_crs, dem_transform
        dem_crs = vv_src.crs
        dem_transform = vv_src.transform

        sar_image = np.sqrt(vv_image ** 2 + vh_image ** 2)
        unwrapped_phase = phase_unwrapping(sar_image)

        dem = (unwrapped_phase - unwrapped_phase.min()) * 255 / (unwrapped_phase.max() - unwrapped_phase.min())
        dem = dem.astype(np.uint8)

        dem_smoothed = gaussian_filter(dem, sigma=1)

        # Dummy NIR and Red bands for NDVI calculation (for demonstration purposes)
        nir = vv_image  # Replace with actual NIR band
        red = vh_image  # Replace with actual Red band
        ndvi = calculate_ndvi(nir, red)

        features = extract_features(vv_image, vh_image, ndvi)
        classified = classify_terrain(features, model)

        apply_colormap_and_save(dem_smoothed, classified, output_path)

        print(f"Generated DEM saved to {output_path}")
    except Exception as e:
        print(f"Error during DEM generation: {e}")
        raise


# Dummy labeled data for training (for demonstration purposes)
# Replace with actual labeled data
X_train = np.random.rand(100, 5)  # 100 samples, 5 features (VV, VH, VV/VH, VV-VH, NDVI)
y_train = np.random.randint(1, 4, 100)  # 3 classes

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

vv_files = get_file_paths(input_dir, '_corrected_VV.tif')
vh_files = get_file_paths(input_dir, '_corrected_VH.tif')

if len(vv_files) != len(vh_files):
    raise ValueError("The number of VV and VH images must be equal.")

for vv_path, vh_path in zip(vv_files, vh_files):
    base_filename = os.path.basename(vv_path).replace('_corrected_VV.tif', '_dem.tif')
    output_path = os.path.join(output_dir, base_filename)
    generate_dem(vv_path, vh_path, output_path, model)
