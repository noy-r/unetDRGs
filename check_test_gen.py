import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

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

            print(
                f"Yielding image: {img_path} with shape: {img.shape}")  # Print the path and shape of the yielded image
            yield img  # Yield the processed image
        except Exception as e:
            print(f"Error reading {img_path}: {e}")  # Print error message if there is an issue reading the image

# Paths and parameters
test_path = 'Processed/test'  # Path to your test images directory
target_size = (256, 256)  # Target size for resizing images
as_gray = False  # Whether to read images as grayscale

# Initialize the generator
generator = testGenerator(test_path, target_size, flag_multi_class=False, as_gray=as_gray)

# Fetch and visualize images from the generator
for img in generator:
    # Plot the image
    plt.figure(figsize=(5, 5))
    if as_gray:
        plt.imshow(np.squeeze(img), cmap='gray')
    else:
        plt.imshow(np.squeeze(img))
    plt.title('Test Image')
    plt.axis('off')
    plt.show()

