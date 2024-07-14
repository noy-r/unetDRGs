import os
import glob
import numpy as np
import skimage.io as io
from check_adjust import adjustData


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=3, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    """
    Generate training data and save as numpy arrays.

    Parameters:
    - image_path: Path to the images.
    - mask_path: Path to the masks.
    - flag_multi_class: Whether to handle multiple classes in masks.
    - num_class: Number of classes.
    - image_prefix: Prefix for image files.
    - mask_prefix: Prefix for mask files.
    - image_as_gray: Whether to read images as grayscale.
    - mask_as_gray: Whether to read masks as grayscale.

    Returns:
    - image_arr: Numpy array of images.
    - mask_arr: Numpy array of masks.
    """
    image_name_arr = glob.glob(os.path.join(image_path, f"{image_prefix}*.png"))
    image_arr = []
    mask_arr = []

    for index, item in enumerate(image_name_arr):
        try:
            # Load and preprocess image
            img = io.imread(item, as_gray=image_as_gray)
            img = np.reshape(img, img.shape + (1,)) if image_as_gray else img

            # Construct the full path to the corresponding mask
            mask_path_full = item.replace(image_path, mask_path).replace(image_prefix, mask_prefix)

            # Check if the mask file exists
            if not os.path.exists(mask_path_full):
                print(f"Mask file not found for image {item}: {mask_path_full}")
                continue

            # Load and preprocess corresponding mask
            mask = io.imread(mask_path_full, as_gray=mask_as_gray)
            mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask

            # Map mask levels to class indices
            mask = np.where(mask == 29, 1, mask)  # Map grayscale level 29 to class 1
            mask = np.where(mask == 150, 2, mask)  # Map grayscale level 150 to class 2

            # Adjust data
            img, mask = adjustData(img, mask, flag_multi_class, num_class)

            # Append to list
            image_arr.append(img)
            mask_arr.append(mask)
        except Exception as e:
            print(f"Error processing {item}: {e}")

    # Convert lists to numpy arrays
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)

    return image_arr, mask_arr


# Example usage
image_path = 'Processed/train/aug/images'
mask_path = 'Processed/train/aug/masks'
image_arr, mask_arr = geneTrainNpy(image_path, mask_path, flag_multi_class=True, num_class=3, image_prefix="image",
                                   mask_prefix="mask", image_as_gray=True, mask_as_gray=True)

print("Images shape:", image_arr.shape)
print("Masks shape:", mask_arr.shape)
