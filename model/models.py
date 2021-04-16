"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense

import hyperparameters as hp


class YourModel(tf.keras.Model):

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam')

        # Conv3x3 (64)
        # Conv3x3 (64)
        # MaxPool
        # Conv3x3 (128)
        # Conv3x3 (128)
        # MaxPool
        # Conv3x3 (256)
        # Conv3x3 (256)
        # Conv3x3 (256)
        # Conv3x3 (256)
        # MaxPool
        # Conv3x3 (512)
        # Conv3x3 (512)
        # Conv3x3 (512)
        # Conv3x3 (512)
        # MaxPool
        # Conv3x3 (512)
        # Conv3x3 (512)
        # Conv3x3 (512)
        # Conv3x3 (512)
        # MaxPool
        # Fully Connected (4096)
        # Fully Connected (4096)
        # Fully Connected (1000)
        # SoftMax

        # Conv2D(64, 3, 1, padding="same",
        #        activation="relu", name="block1_conv1"),
        # Conv2D(64, 3, 1, padding="same",
        #        activation="relu", name="block1_conv2"),
        # MaxPool2D(2, name="block1_pool"),
        # # Block 2
        # Conv2D(128, 3, 1, padding="same",
        #        activation="relu", name="block2_conv1"),
        # Conv2D(128, 3, 1, padding="same",
        #        activation="relu", name="block2_conv2"),
        # MaxPool2D(2, name="block2_pool"),
        # # Block 3
        # Conv2D(256, 3, 1, padding="same",
        #        activation="relu", name="block3_conv1"),
        # Conv2D(256, 3, 1, padding="same",
        #        activation="relu", name="block3_conv2"),
        # Conv2D(256, 3, 1, padding="same",
        #        activation="relu", name="block3_conv3"),
        # MaxPool2D(2, name="block3_pool"),
        # # Block 4
        # Conv2D(512, 3, 1, padding="same",
        #        activation="relu", name="block4_conv1"),
        # Conv2D(512, 3, 1, padding="same",
        #        activation="relu", name="block4_conv2"),
        # Conv2D(512, 3, 1, padding="same",
        #        activation="relu", name="block4_conv3"),
        # MaxPool2D(2, name="block4_pool"),
        # # Block 5
        # Conv2D(512, 3, 1, padding="same",
        #        activation="relu", name="block5_conv1"),
        # Conv2D(512, 3, 1, padding="same",
        #        activation="relu", name="block5_conv2"),
        # Conv2D(512, 3, 1, padding="same",
        #        activation="relu", name="block5_conv3"),
        # MaxPool2D(2, name="block5_pool"),
        self.architecture = [
            Conv2D(64, (3, 3), 1, activation='relu', padding='same', input_shape=(hp.img_size, hp.img_size, 3)),
            Conv2D(64, (3, 3), 1, activation='relu', padding='same'),
            MaxPool2D((2, 2), 2),
            Conv2D(128, (3, 3), 1, activation='relu', padding='same'),
            Conv2D(128, (3, 3), 1, activation='relu', padding='same'),
            MaxPool2D((2, 2), 2),
            Conv2D(256, (3, 3), 1, activation='relu', padding='same'),
            Conv2D(256, (3, 3), 1, activation='relu', padding='same'),
            MaxPool2D((2, 2), 2),
            Conv2D(512, (3, 3), 1, activation='relu', padding='same'),
            Conv2D(512, (3, 3), 1, activation='relu', padding='same'),
            MaxPool2D((2, 2), 2),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.4),
            Dense(2048, activation='relu'),
            Dropout(0.4),
            Dense(1000, activation='relu'),
            Dense(26, activation='softmax')
        ]

        # Conv2D(128, (5, 5), 2, activation='relu', padding='valid', input_shape=(hp.img_size, hp.img_size, 3)),
        # # 205 - 5, / 2 = 100 + 1
        # Conv2D(128, (3, 3), 2, activation='relu', padding='valid', input_shape=(101, 101, 128)),
        # # 101 - 3, / 2 = 49 + 1 = 50
        # MaxPool2D((2, 2), 2, padding="valid"),
        # Dropout(0.25),
        #
        # # 50 - 2, / 2 + 1 = 25
        # Conv2D(128, (3, 3), 2, activation='relu', padding='valid', input_shape=(25, 25, 64)),
        # # 25 - 3, / 2 = 11 + 1 = 12
        # Conv2D(128, (3, 3), 1, activation='relu', padding='valid', input_shape=(12, 12, 128)),
        # # 12 - 3 = 9 + 1 = 10
        # MaxPool2D((2, 2), 1, padding="valid"),
        # Dropout(0.25),
        #
        # # 12 - 2 + 1 = 11
        # Conv2D(256, (3, 3), 1, activation='relu', padding="same", input_shape=(11, 11, 128)),
        # Conv2D(256, (3, 3), 1, activation='relu', padding="same", input_shape=(11, 11, 256)),
        # MaxPool2D((2, 2), 1, padding="valid"),
        # Dropout(0.2),
        #
        # Flatten(),
        # Dense(512, activation='relu'),
        # Dropout(0.4),
        # Dense(256, activation='relu'),
        # Dropout(0.2),
        # Dense(256, activation='relu'),
        # Dense(26, activation='softmax')

    def call(self, x):
        """ Passes input image through the network. """
        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
