import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.constraints import MaxNorm


# Define the custom loss function with regularization
def custom_loss(y_true, y_pred, model, lambda_reg=1e-4):
    # Calculate the categorical cross-entropy loss
    cce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Calculate the regularization term
    reg_term = tf.reduce_sum([tf.nn.l2_loss(v) for v in model.trainable_variables])

    # Combine the two
    total_loss = cce_loss + lambda_reg * reg_term
    return total_loss


def unet(pretrained_weights=None, input_size=(256, 256, 3), num_classes=3, lambda_reg=1e-4):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_constraint=MaxNorm(3))(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_constraint=MaxNorm(3))(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_constraint=MaxNorm(3))(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_constraint=MaxNorm(3))(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_constraint=MaxNorm(3))(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)  # Use 'softmax' for multi-class segmentation

    model = Model(inputs=inputs, outputs=conv10)

    # Use the custom loss function with regularization
    def loss_fn(y_true, y_pred):
        return custom_loss(y_true, y_pred, model, lambda_reg)

    model.compile(optimizer=Adam(learning_rate=1e-5, clipvalue=1.0), loss=loss_fn, metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
