import numpy as np


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


# Define your custom color mappings for visualization
Class1 = [0, 0, 255]  # Blue
Class2 = [0, 255, 0]  # Green
Background = [0, 0, 0]  # Black

COLOR_DICT = np.array([Background, Class1, Class2])

# Example usage
num_class = 3
labeled_img = np.array([[0, 1, 2], [2, 1, 0]])  # Example labeled image

visualized_img = labelVisualize(num_class, COLOR_DICT, labeled_img)
print(visualized_img)


import os
import numpy as np
import skimage.io as io
from skimage import img_as_ubyte


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=3):
    """
    Save prediction results as images.

    Parameters:
    - save_path: Directory where the images will be saved.
    - npyfile: Numpy array of predicted masks.
    - flag_multi_class: Whether the masks are multi-class.
    - num_class: Number of classes.
    """
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, item in enumerate(npyfile):
        if flag_multi_class:
            # Visualize the multi-class mask
            img = labelVisualize(num_class, COLOR_DICT, item)
        else:
            # Use the single-channel mask
            img = item[:, :, 0]

        # Normalize the image to 8-bit
        img = img_as_ubyte(img)

        # Save the image
        save_path_full = os.path.join(save_path, f"{i}_predict.png")
        io.imsave(save_path_full, img)


# Example usage
if __name__ == "__main__":
    # Example npyfile with dummy data
    dummy_npyfile = np.random.randint(0, 3, (5, 256, 256, 1))  # Example prediction array
    save_path = 'output'  # Directory to save results

    # Save the results
    saveResult(save_path, dummy_npyfile, flag_multi_class=True, num_class=3)
    print(f"Saved predictions to {save_path}")
