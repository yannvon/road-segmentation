from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

import constants
from preprocessing_helper import *


class BasicModel:
    """ A simple basic model following the provided template """

    def __init__(self):

        # create model
        self.model = Sequential()

        # add model layers
        self.model.add(Conv2D(16, kernel_size=constants.IMG_PATCH_SIZE, activation='relu',
                              input_shape=(constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE, constants.NUM_CHANNELS)))
        self.model.add(Flatten())
        self.model.add(Dense(2, activation='softmax'))

        # compile model using accuracy to measure model performance
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data, train_labels, epochs=100):

        # Step 1: Either put a class weight or remove some data from train set to have 50 / 50 data per class
        X, Y = get_equal_train_set_per_class(train_data, train_labels)

        # Step 2: Other things

        # Finally, train the model !
        self.model.fit(train_data, train_labels, epochs)

    def predict(self, data):
        return self.model.predict(data)