from model import unet
from data import trainGenerator, testGenerator, saveResult
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Define augmentation parameters
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

# Define paths
train_path = 'Processed/train'
image_folder = 'images'
mask_folder = 'masks'
test_path = 'Processed/test'

# Create data generator for training
batch_size = 1
num_classes = 3

myGene = trainGenerator(batch_size, train_path, image_folder, mask_folder, data_gen_args,
                        image_color_mode="rgb", mask_color_mode="grayscale",
                        flag_multi_class=True, num_class=num_classes, save_to_dir=None)

# Create the UNet model
model = unet(input_size=(256, 256, 3), num_classes=num_classes)

# Define model checkpoints
model_checkpoint = ModelCheckpoint('unet_neuron.keras', monitor='loss', verbose=1, save_best_only=True)

# Function to plot images
def plot_images(img, prediction, mask, iteration):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img[0])
    ax[0].set_title('Input Image')
    ax[1].imshow(np.argmax(prediction[0], axis=-1))
    ax[1].set_title('Prediction')
    ax[2].imshow(np.argmax(mask[0], axis=-1))
    ax[2].set_title('Ground Truth')
    plt.suptitle(f'Iteration {iteration}')
    plt.show()

# Train the model and print images every 20 iterations
steps_per_epoch = 200
epochs = 5

# Lists to store loss and accuracy values
losses = []
accuracies = []

@tf.function
def train_step(model, img, mask):
    with tf.GradientTape() as tape:
        prediction = model(img, training=True)
        loss = tf.keras.losses.categorical_crossentropy(mask, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(mask, prediction))
    return loss, accuracy

for epoch in range(epochs):
    for iteration in range(steps_per_epoch):
        img, mask = next(myGene)
        loss, accuracy = train_step(model, img, mask)
        losses.append(loss.numpy().mean())
        accuracies.append(accuracy.numpy().mean())
        print(f'Epoch {epoch + 1}, Iteration {iteration + 1}, Loss: {loss.numpy().mean()}, Accuracy: {accuracy.numpy().mean()}')
        if (iteration + 1) % 20 == 0:
            prediction = model.predict(img)
            plot_images(img, prediction, mask, iteration + 1)
    model.save(f'unet_epoch_{epoch + 1}.keras')

# Plot the loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy over Iterations')
plt.legend()

plt.tight_layout()
plt.show()

# Create data generator for testing
testGene = testGenerator(test_path, target_size=(256, 256), flag_multi_class=True, as_gray=False)

# Check if generator yields data
print("Checking test generator output...")
test_imgs = list(testGene)  # Convert generator to list to debug
print(f"Test images loaded: {len(test_imgs)}")

# Predict results on test data if images are found
if test_imgs:
    results = model.predict(np.vstack(test_imgs), verbose=1)
    # Ensure the results directory exists
    results_dir = os.path.join(test_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    # Save the results
    saveResult(results_dir, results, flag_multi_class=True, num_class=num_classes)
else:
    print("No test images found.")

print("Training and testing complete.")
