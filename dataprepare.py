import os
import matplotlib.pyplot as plt
import numpy as np
from data import *
import cv2

# Define augmentation parameters
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

# Create directories for augmented data if they don't exist
aug_dir = "Processed/train/aug"
os.makedirs(os.path.join(aug_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(aug_dir, 'masks'), exist_ok=True)

def map_to_nearest_level(mask, original_levels):
    """Map mask values to the nearest original level."""
    original_levels = np.array(original_levels)
    reshaped_mask = mask.reshape(-1)
    mapped_mask = np.zeros_like(reshaped_mask)

    for i, value in enumerate(reshaped_mask):
        mapped_mask[i] = original_levels[np.argmin(np.abs(original_levels - value))]

    return mapped_mask.reshape(mask.shape)

# Create a data generator
myGenerator = trainGenerator(
    20,
    'Processed/train',  # Path to the training data directory
    'images',           # Subdirectory with images
    'masks',            # Subdirectory with masks
    data_gen_args,
    save_to_dir=aug_dir  # Directory to save augmented images and masks
)

# Generate and save a few batches of augmented data
num_batch = 3
for i, batch in enumerate(myGenerator):
    print(f"Processing batch {i + 1}")
    if i >= num_batch:
        break

# Load the augmented data as numpy arrays
print("Loading augmented data as numpy arrays...")
image_arr, mask_arr = geneTrainNpy(os.path.join(aug_dir, 'images'),
                                  os.path.join(aug_dir, 'masks'),
                                  flag_multi_class=True, num_class=3, image_as_gray=False)

if image_arr is not None and mask_arr is not None:
    # Optionally save the numpy arrays
    np.save("Processed/image_arr.npy", image_arr)
    np.save("Processed/mask_arr.npy", mask_arr)

    # Inspect the augmented data
    print(f"Image array shape: {image_arr.shape}")
    print(f"Mask array shape: {mask_arr.shape}")

    # Display a few examples
    num_examples = 3
    fig, ax = plt.subplots(num_examples, 2, figsize=(12, 12))
    for i in range(num_examples):
        ax[i, 0].imshow(image_arr[i])
        ax[i, 0].set_title(f"Augmented Image {i+1}")

        # Collapse the one-hot encoded mask into a single channel with 3 grayscale levels
        mask = np.argmax(mask_arr[i], axis=-1).astype(float)
        mask = (mask / 2) * 255  # Rescale the values to 0-255 range
        ax[i, 1].imshow(mask, cmap='gray')
        ax[i, 1].set_title(f"Augmented Mask {i+1}")

    plt.show()

    # Check the unique grayscale levels in the masks
    print("Checking unique grayscale levels in the masks...")
    for i in range(num_examples):
        unique_values = np.unique(mask_arr[i])
        print(f"Unique values in Mask {i+1}: {unique_values}")
else:
    print("No data available to display.")
