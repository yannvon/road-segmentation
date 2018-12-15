from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import model_from_json
from sklearn.metrics import f1_score
import numpy as np

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.regularizers import l2

# FIXME too many imports, remove unnecessary
from VGG import VGGModel
from keras import callbacks
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam


import constants
from preprocessing_helper import *
from postprocessing_helper import *


class YannModel:
    """ A simple model inspired by the VGG model """
    
    WINDOW_SIZE = 80
    OUTPUT_FILENAME = "yann_model"

    def __init__(self):

        _input = Input((80,80,3)) 

        conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
        conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
        pool1  = MaxPooling2D((2, 2))(conv2)

        conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
        conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
        pool2  = MaxPooling2D((2, 2))(conv4)

        conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
        conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
        conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
        pool3  = MaxPooling2D((2, 2))(conv7)

        conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
        conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
        conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
        pool4  = MaxPooling2D((2, 2))(conv10)

        conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
        conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
        conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
        pool5  = MaxPooling2D((2, 2))(conv13)

        flat   = Flatten()(pool5)
        dense1 = Dense(4096, activation="relu")(flat)
        dense2 = Dense(4096, activation="relu")(dense1)
        output = Dense(1000, activation="softmax")(dense2)
        classification = Dense(2, activation="softmax")(output)
        
        
        vgg16_model  = Model(inputs=_input, outputs=classification)          
        
        self.model = vgg16_model
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        
        
    def load_data(self, train_data_filename, train_labels_filename, training_size):
        # Extract it into numpy arrays.
        self.train_data = extract_data(train_data_filename, training_size, self.WINDOW_SIZE)
        self.train_labels = extract_labels(train_labels_filename, training_size)
        
        

    def train(self, epochs=30, validation_split=0.2):
        
        # Step 0: Shuffle samples
        np.random.seed(0)
        np.random.shuffle(self.train_data)
        # resetting the seed allows for an identical shuffling between y and x
        np.random.seed(0)
        np.random.shuffle(self.train_labels)

        # Step 1: Split into validation and training set     
        split_index = int(len(self.train_data) * (1 - validation_split))
        self.train_data_split = self.train_data[0:split_index]
        self.validation_data_split = self.train_data[split_index:len(self.train_data)]
        self.train_label_split = self.train_labels[0:split_index]
        self.validation_label_split = self.train_labels[split_index:len(self.train_data)]
        
        # Step 2: Give weights to classes
        c_weight = {1: 2.8, 
                    0: 1.}
        
        # (depracated) Step 2 : other option, cut out data !
        # X, Y = get_equal_train_set_per_class(train_data, train_labels)

        # Step 3: Greate Generators
        train_datagen = ImageDataGenerator(
            #rotation_range=180,
            horizontal_flip=True,
            vertical_flip=True)

        validation_datagen = ImageDataGenerator()
            #rotation_range=180,
            #horizontal_flip=True,
            #vertical_flip=True)
            
        train_generator = train_datagen.flow(self.train_data_split, self.train_label_split, batch_size=32)
        validation_generator = validation_datagen.flow(self.validation_data_split, 
                                                       self.validation_label_split, batch_size=32)

        
        # Step 4: Early stop
        early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, 
                                            mode='max', restore_best_weights=True)

        # Finally, train the model !
        # Training
        self.model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=len(self.train_data_split)/32,
                    epochs=epochs,
                    callbacks = [early_stop_callback],
                    class_weight=c_weight,
                    validation_steps=len(self.validation_data_split)/16,
                    use_multiprocessing=True)
        
    
    def generate_submission(self):
        createSubmission(self.model, self.WINDOW_SIZE)
        
            
    def generate_images(self, prediction_training_dir, train_data_filename):
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        
        for i in range(1, constants.TRAINING_SIZE+1):
            pimg = get_prediction_with_mask(train_data_filename, i, self.model, self.WINDOW_SIZE)
            Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
            oimg = get_prediction_with_overlay(train_data_filename, i, self.model, self.WINDOW_SIZE)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        
  
    def save(self):
        # save model on disk
        # source https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.OUTPUT_FILENAME + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.OUTPUT_FILENAME + ".h5")
        print("Saved model to disk")
        
    def load(self):
        pass
        # FIXME
        ##load json and create model
        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #loaded_model = model_from_json(loaded_model_json)
        ##load weights into new model
        #loaded_model.load_weights("model.h5")
        #loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1])
        #print("Loaded model from disk")
        
    def f1_score(self):
        validation_data_prediction = self.model.predict_classes(self.validation_data_split)
        validation_label = []
        for e in self.validation_label_split:
            if (e[0] == 0):
                validation_label.append(1)
            else:
                validation_label.append(0)
        validation_label = np.array(validation_label)     
        print("F1 score = "+str(f1_score(validation_data_prediction, validation_label, labels=['1'])))
