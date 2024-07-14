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
        mask = ensure_levels(mask, original_levels)  # Ensure all required levels are present
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


# Paths and parameters
train_path = 'Processed/train'
image_folder = 'images'
mask_folder = 'masks'
aug_dict = dict(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                horizontal_flip=True, fill_mode='nearest')

# Directory to save augmented images and masks
save_to_dir = 'Processed/train/aug'

# Ensure the directory exists
ensure_dir_exists(save_to_dir)
ensure_dir_exists(os.path.join(save_to_dir, 'images'))
ensure_dir_exists(os.path.join(save_to_dir, 'masks'))

# Initialize the generator
generator = trainGenerator(
    batch_size=1,
    train_path=train_path,
    image_folder=image_folder,
    mask_folder=mask_folder,
    aug_dict=aug_dict,
    flag_multi_class=True,
    num_class=3,
    target_size=(256, 256),
    save_to_dir=save_to_dir
)

# Generate multiple batches to save more images and masks
num_batches = 20  # Number of batches to generate
for batch_index in range(num_batches):
    img, mask = next(generator)
    print(f"Batch {batch_index}: Image shape: {img.shape}, Mask shape: {mask.shape}")

    # Plotting the images for verification
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img[0])
    axes[0].set_title('Image')
    axes[1].imshow(mask[0, :, :, 0], cmap='gray')
    axes[1].set_title('Mask Channel 1')
    axes[2].imshow(mask[0, :, :, 1], cmap='gray')
    axes[2].set_title('Mask Channel 2')
    axes[3].imshow(mask[0, :, :, 2], cmap='gray')
    axes[3].set_title('Mask Channel 3')

    for ax in axes:
        ax.axis('off')

    plt.show()

print("Generated images and masks with proper grey levels.")
