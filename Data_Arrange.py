import os
import shutil

# Paths to your original images directory
original_dir = '/Users/noymachluf/Desktop/Unet_DRG/Segmented_original'
images_dir = '/Users/noymachluf/Desktop/Unet_DRG/Images'
masks_dir = '/Users/noymachluf/Desktop/Unet_DRG/Masks'

# Create new directories if they do not exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# Get list of files in the original directory
files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

# Initialize counters
count = 1

# Create a dictionary to pair images and masks
paired_files = {}

# Pair images and masks based on common prefixes
for file_name in files:
    if file_name.endswith('.png'):
        # Remove the suffix to find the common prefix
        if 'original' in file_name:
            prefix = file_name.replace('_original.png', '')
            paired_files.setdefault(prefix, {})['image'] = file_name
        elif 'nuclei_and_axons_masks' in file_name:
            prefix = file_name.replace('_nuclei_and_axons_masks.png', '')
            paired_files.setdefault(prefix, {})['mask'] = file_name

# Move and rename files
for prefix, pair in paired_files.items():
    if 'image' in pair and 'mask' in pair:
        image_file = pair['image']
        mask_file = pair['mask']

        new_image_name = f'image{count}.png'
        new_mask_name = f'mask{count}.png'

        shutil.copy(os.path.join(original_dir, image_file), os.path.join(images_dir, new_image_name))
        shutil.copy(os.path.join(original_dir, mask_file), os.path.join(masks_dir, new_mask_name))

        print(f'Copied {image_file} to {new_image_name} and {mask_file} to {new_mask_name}')

        count += 1

print(f'Moved and renamed {count - 1} pairs of images and masks to the new directories.')
