import numpy as np
import matplotlib.pyplot as plt
import cv2

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

# Paths to your image and mask
image_path = 'Processed/train/images/image3.png'
mask_path = 'Processed/train/masks/mask3.png'

# Load the image and mask
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Check the output of the adjustData function
adjusted_image, adjusted_mask = adjustData(image, mask, flag_multi_class=True, num_class=3)

# Plotting the images for verification
fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # Create six subplots
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Original Mask')
axes[2].imshow(adjusted_image)
axes[2].set_title('Adjusted Image')

# Plot each channel of the adjusted mask
for i in range(3):
    axes[3 + i].imshow(adjusted_mask[:, :, i], cmap='gray')
    axes[3 + i].set_title(f'Adjusted Mask Channel {i+1}')

for ax in axes:
    ax.axis('off')

plt.show()

# Reload the mask to display unique grey levels
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Display the mask
plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap='gray')
plt.title('Original Mask')
plt.axis('off')
plt.show()

# Print the unique grey levels in the mask
unique_grey_levels = np.unique(mask)
print("Unique grey levels in the mask:", unique_grey_levels)
