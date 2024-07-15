import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# Define paths
model_path = 'unet_epoch_6.keras'  # Path to the saved model

def predict_single_image(image_path, model_path, target_size=(256, 256)):
    # Load the pre-trained model
    model = load_model(model_path)

    # Load and preprocess the image
    original_img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(original_img)
    img_array = img_array / 255.0  # Normalization step if your model expects this normalization
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for the model (adding batch dimension)

    # Predict the mask
    prediction = model.predict(img_array)
    predicted_mask = np.argmax(prediction[0], axis=-1)  # Assuming the model outputs class probabilities

    # Plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_img)
    ax[0].set_title('Original Image')
    ax[1].imshow(predicted_mask, cmap='gray')  # Change colormap if needed
    ax[1].set_title('Predicted Mask')
    plt.show()

# Example usage
image_path = 'Images/image3.png'  # Change to your image path
predict_single_image(image_path, model_path)
