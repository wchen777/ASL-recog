import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import base64
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

model_path = 'ASL-classification-model.h5'
vgg_path = 'vgg16_imagenet.h5'


class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam')

        # Don't change the below:

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1", trainable=False),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2", trainable=False),
            MaxPool2D(2, name="block1_pool", trainable=False),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1", trainable=False),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2", trainable=False),
            MaxPool2D(2, name="block2_pool", trainable=False),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1", trainable=False),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2", trainable=False),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3", trainable=False),
            MaxPool2D(2, name="block3_pool", trainable=False),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1", trainable=False),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2", trainable=False),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3", trainable=False),
            MaxPool2D(2, name="block4_pool", trainable=False),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1", trainable=False),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2", trainable=False),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3", trainable=False),
            MaxPool2D(2, name="block5_pool", trainable=False)
        ]

        self.head = [Flatten(),
                     Dense(1024, activation='relu'),
                     Dropout(0.4),
                     Dense(1024, activation='relu'),
                     Dropout(0.4),
                     Dense(512, activation='relu'),
                     Dense(26, activation='softmax')]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """
        x = self.vgg16(x)
        x = self.head(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)


# initialize model
def init_model():
    model = VGGModel()
    model(tf.keras.Input(shape=(224, 224, 3)))
    model.vgg16.load_weights(vgg_path, by_name=True)
    model.head.load_weights(model_path, by_name=False)
    print("loaded model successfully")
    return model


# decode from base64 and save image
def process_image(img_url):
    with open('img_output.png', 'wb') as output:
        output.write(base64.b64decode(img_url))


def predict(model):
    img_data = cv2.imread('img_output.png')
    img_resize = cv2.resize(img_data, (224, 224))
    img_resize = preprocess_input(img_resize)
    predictions = model.predict(np.array([img_resize]))
    output = predictions[0].argsort()[-3:][::-1]
    return output
