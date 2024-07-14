import os
import numpy as np
import skimage.io as io


def check_mask_levels(mask_path, num_images=None):
    mask_files = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.png')]
    if num_images is not None:
        mask_files = mask_files[:num_images]

    unique_values = set()
    for mask_file in mask_files:
        mask = io.imread(mask_file, as_gray=True)
        unique_values.update(np.unique(mask))

    print("Unique grayscale levels in masks:", sorted(unique_values))


# Path to your mask directory
mask_dir = 'Processed/train/aug/masks'  # Change this to the directory where your masks are stored

# Check grayscale levels in masks
check_mask_levels(mask_dir, num_images=10)  # Adjust num_images to check a specific number of masks
