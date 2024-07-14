import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from keras.models import load_model
from skimage import img_as_ubyte

# Define paths
test_path = 'Processed/test'
model_path = 'unet_epoch_6.keras'  # Path to the saved model

# Function to generate test images and masks
def testGenerator(test_path, target_size=(256, 256), flag_multi_class=True, as_gray=False):
    image_path = os.path.join(test_path, 'images')  # Path to test images
    mask_path = os.path.join(test_path, 'masks')  # Path to test masks
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # Supported image file extensions
    image_files = sorted([f for f in os.listdir(image_path) if f.lower().endswith(image_extensions)])  # List of test image files
    mask_files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith(image_extensions)])  # List of test mask files

    print(f"Found {len(image_files)} images in {image_path}")  # Print number of found test images

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_path, img_file)  # Full path to the image file
        mask_path_full = os.path.join(mask_path, mask_file)  # Full path to the mask file
        try:
            img = io.imread(img_path, as_gray=as_gray)  # Read the image
            img = img / 255.0  # Normalize the image
            img = transform.resize(img, target_size, mode='constant', preserve_range=True)  # Resize the image

            mask = io.imread(mask_path_full, as_gray=True)  # Read the mask
            mask = mask / 255.0  # Normalize the mask
            mask = transform.resize(mask, target_size, mode='constant', preserve_range=True)  # Resize the mask

            if as_gray:
                img = np.expand_dims(img, axis=-1)  # Add channel dimension if grayscale
                mask = np.expand_dims(mask, axis=-1)  # Add channel dimension if grayscale
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            mask = np.expand_dims(mask, axis=0)  # Add batch dimension

            print(f"Yielding image: {img_path} with shape: {img.shape}, mask shape: {mask.shape}")  # Print the path and shape of the yielded image and mask
            yield img, mask  # Yield the processed image and mask
        except Exception as e:
            print(f"Error reading {img_path} or {mask_path_full}: {e}")  # Print error message if there is an issue reading the image

# Function to save the predicted results
def saveResult(save_path, npyfile, flag_multi_class=True, num_class=3):
    for i, item in enumerate(npyfile):
        img = np.argmax(item, axis=-1)
        img = img.astype(np.uint8)
        save_path_full = os.path.join(save_path, f"{i}_predict.png")
        io.imsave(save_path_full, img_as_ubyte(img / np.max(img)))  # Ensure proper scaling

# Load the pre-trained model
model = load_model(model_path)

# Create data generator for testing
testGene = testGenerator(test_path, target_size=(256, 256), flag_multi_class=True, as_gray=False)

# Check if generator yields data
print("Checking test generator output...")
test_data = list(testGene)  # Convert generator to list to debug
test_imgs, test_masks = zip(*test_data)  # Separate images and masks
print(f"Test images loaded: {len(test_imgs)}")

# Predict results on test data if images are found
if test_imgs:
    results = model.predict(np.vstack(test_imgs), verbose=1)
    # Ensure the results directory exists
    results_dir = os.path.join(test_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    # Save the results
    saveResult(results_dir, results, flag_multi_class=True, num_class=3)

    # Visualize multiple examples
    num_examples = 5  # Number of examples to plot
    for i in range(num_examples):
        example_img = test_imgs[i]
        example_mask = test_masks[i]
        example_prediction = results[i]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(example_img[0])
        ax[0].set_title('Test Image')
        ax[1].imshow(example_mask[0], cmap='gray')  # Display the mask properly
        ax[1].set_title('Ground Truth Mask')
        ax[2].imshow(np.argmax(example_prediction, axis=-1))
        ax[2].set_title('Predicted Mask')
        plt.show()
else:
    print("No test images found.")

print("Testing and saving predictions complete.")
