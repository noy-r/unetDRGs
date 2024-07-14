# Image Segmentation using U-Net

This project implements image segmentation using the U-Net architecture in Python with Keras. The model is trained on a custom dataset and can segment images into multiple classes.

## Project Structure

The project consists of the following files:

- `data.py`: Contains utility functions for data preprocessing, augmentation, and generation.
- `dataprepare.py`: Prepares the training data by applying data augmentation and saving the augmented data as numpy arrays.
- `model.py`: Defines the U-Net model architecture.
- `trainUnet.py`: Trains the U-Net model on the prepared training data and saves the best model.
- `main.py`: Loads the trained model, generates predictions on the test data, and visualizes the results.

## Dataset

The dataset should be organized in the following structure:

```
Processed/
├── train/
│   ├── images/
│   └── masks/
└── test/
    └── images/
    └── masks/
```

- The `train` directory contains the training images and their corresponding masks.
- The `test` directory contains the test images for evaluation.

## Usage

1. Prepare your dataset in the required structure.

2. Run `dataprepare.py` to apply data augmentation and save the augmented data as numpy arrays:
   ```
   python dataprepare.py
   ```

3. Train the U-Net model by running `trainUnet.py`:
   ```
   python trainUnet.py
   ```
   The script will train the model for a specified number of epochs and save the best model based on the validation loss.

4. Use the trained model for prediction and visualization by running `main.py`:
   ```
   python main.py
   ```
   The script will load the trained model, generate predictions on the test images, save the predicted masks, and visualize some examples.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- Keras
- TensorFlow
- NumPy
- Matplotlib
- scikit-image

You can install the required packages using pip:
```
pip install keras tensorflow numpy matplotlib scikit-image
```

## Customization

You can customize various aspects of the project:

- Modify the `data_gen_args` dictionary in `dataprepare.py` and `main.py` to adjust the data augmentation parameters.
- Change the `batch_size`, `steps_per_epoch`, and `epochs` variables in `trainUnet.py` to control the training process.
- Adjust the `input_size` and `num_classes` parameters when creating the U-Net model in `model.py` and `main.py` based on your dataset.

## Results

The predicted masks will be saved in the `Processed/test/results` directory. The script will also visualize some examples of the test images, ground truth masks, and predicted masks.

## License

This project is licensed under the [MIT License](LICENSE).
