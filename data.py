from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
from skimage import img_as_ubyte
import cv2
import skimage.transform as transform


# Define your custom color mappings for visualization
Class1 = [0, 0, 255]  # Blue
Class2 = [0, 255, 0]  # Green
Background = [0, 0, 0]  # Black

COLOR_DICT = np.array([Background, Class1, Class2])


# Function to adjust image and mask data
def adjustData(img, mask, flag_multi_class, num_class):
    img = img / 255.0  # Normalize image data to range [0, 1]

    if flag_multi_class:
        # Adjust mask levels based on unique grey levels found in the mask
        mask_levels = np.unique(mask)
        print(f"Unique mask levels: {mask_levels}")

        if len(mask_levels) != num_class:
            raise ValueError(f"Number of unique mask levels ({len(mask_levels)}) does not match num_class ({num_class})")

        new_mask = np.zeros(mask.shape + (num_class,))  # Initialize a new mask with an extra dimension for one-hot encoding
        for i, level in enumerate(mask_levels):
            new_mask[mask == level, i] = 1  # One-hot encode the mask
        mask = new_mask
    else:
        mask = mask / 255.0  # Normalize mask data to range [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Expand dimensions to add a channel dimension
        mask = np.where(mask > 0.5, 1, 0)  # Binarize the mask

    print(f"Adjusted image shape: {img.shape}, Adjusted mask shape: {mask.shape}")
    return img, mask


import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
from skimage import img_as_ubyte
import cv2
from check_adjust import adjustData


def map_to_nearest_level(mask, original_levels):
    """Map mask values to the nearest original level."""
    original_levels = np.array(original_levels)
    reshaped_mask = mask.reshape(-1)
    mapped_mask = np.zeros_like(reshaped_mask)

    for i, value in enumerate(reshaped_mask):
        mapped_mask[i] = original_levels[np.argmin(np.abs(original_levels - value))]

    return mapped_mask.reshape(mask.shape)


def clean_mask_channel_2(mask, kernel_size=9):
    """Remove small artifacts from channel 2 of the mask using morphological operations."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    class_mask = mask[..., 1].astype(np.uint8)  # Apply to channel 2 (index 1)
    cleaned_class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
    mask[..., 1] = cleaned_class_mask
    return mask


def ensure_dir_exists(directory):
    """Ensure that a directory exists; if it does not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_mask(mask, filepath, original_levels=[0, 29, 150]):
    """Save mask with correct levels."""
    mapped_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i, level in enumerate(original_levels):
        mapped_mask[mask[..., i] == 1] = level
    io.imsave(filepath, img_as_ubyte(mapped_mask))


def ensure_levels(mask, required_levels):
    """Ensure all required levels are present in the mask."""
    present_levels = np.unique(mask)
    for level in required_levels:
        if level not in present_levels:
            mask[0, 0] = level  # Reintroduce missing levels
    return mask

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=True, num_class=3, save_to_dir=None, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=None,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)
    original_levels = [0, 29, 150]  # Original levels in the mask

    for i, (img, mask) in enumerate(train_generator):
        mask = map_to_nearest_level(mask, original_levels)  # Map mask values to the nearest original level
        unique_levels = np.unique(mask)
        if len(unique_levels) < num_class:
            missing_levels = [level for level in original_levels if level not in unique_levels]
            for level in missing_levels:
                mask[0, 0, 0] = level  # Assign the missing level to a pixel (to ensure all levels are present)
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        mask = mask.squeeze(axis=-2)  # Remove the singleton dimension if it exists
        mask = clean_mask_channel_2(mask, kernel_size=9)  # Clean the mask on channel 2 only with kernel size 9

        # Save the augmented images and masks if save_to_dir is specified
        if save_to_dir:
            image_path = os.path.join(save_to_dir, 'images', f"{image_save_prefix}_{i}.png")
            mask_path = os.path.join(save_to_dir, 'masks', f"{mask_save_prefix}_{i}.png")
            io.imsave(image_path, img_as_ubyte(img[0]))
            save_mask(mask[0], mask_path)

        yield (img, mask)

def testGenerator(test_path, target_size=(256, 256), flag_multi_class=True, as_gray=False):
    image_path = os.path.join(test_path, 'images')  # Path to test images
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # Supported image file extensions
    image_files = sorted(
        [f for f in os.listdir(image_path) if f.lower().endswith(image_extensions)])  # List of test image files

    print(f"Found {len(image_files)} images in {image_path}")  # Print number of found test images

    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)  # Full path to the image file
        try:
            img = io.imread(img_path, as_gray=as_gray)  # Read the image
            img = img / 255.0  # Normalize the image
            img = transform.resize(img, target_size, mode='constant', preserve_range=True)  # Resize the image

            if as_gray:
                img = np.expand_dims(img, axis=-1)  # Add channel dimension if grayscale
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            print(f"Yielding image: {img_path} with shape: {img.shape}")  # Print the path and shape of the yielded image
            yield img  # Yield the processed image
        except Exception as e:
            print(f"Error reading {img_path}: {e}")  # Print error message if there is an issue reading the image


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


def labelVisualize(num_class, color_dict, img):
    """
    Visualize the labeled image by mapping class indices to colors.

    Parameters:
    - num_class: Number of classes.
    - color_dict: Dictionary mapping class indices to RGB colors.
    - img: Labeled image with class indices.

    Returns:
    - img_out: RGB image with colors representing class indices.
    """
    # Ensure the image has the correct dimensions
    img = img[:, :, 0] if len(img.shape) == 3 and img.shape[2] == 1 else img

    # Initialize the output image with the same height and width as img and 3 color channels
    img_out = np.zeros(img.shape + (3,))

    # Validate the color dictionary length
    if len(color_dict) != num_class:
        raise ValueError("The length of color_dict does not match num_class.")

    # Map class indices to colors
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]

    return img_out / 255



import os
import numpy as np
from skimage import io, img_as_ubyte

# Function to save the predicted results
def saveResult(save_path, npyfile, flag_multi_class=True, num_class=3):
    for i, item in enumerate(npyfile):
        img = np.argmax(item, axis=-1)
        img = img.astype(np.uint8)
        save_path_full = os.path.join(save_path, f"{i}_predict.png")
        io.imsave(save_path_full, img_as_ubyte(img / np.max(img)))  # Ensure proper scaling
