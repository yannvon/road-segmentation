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


class VGGModel2:
    """ A simple model inspired by the VGG model """
    
    WINDOW_SIZE = 80
    OUTPUT_FILENAME = "vgg_model"

    def __init__(self):

        ##Parameters
        model = Sequential()
        kernel_size = (3,3)
        pool_size = (2,2)
        alpha_relu = 0.1
        regularizer = 1e-6

        #Size of input matrix
        #To change according to the shape
        shape = (self.WINDOW_SIZE, self.WINDOW_SIZE, 3)
        model = Sequential()


        #ITERATION 1

        #Add convolution 
        model.add(Convolution2D(64,
                                kernel_size,
                                padding='same',
                                input_shape=shape))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))
        model.add(Convolution2D(64,
                                kernel_size,
                                padding='same',
                                input_shape=shape))
        model.add(Dropout(0.1))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))

        model.add(Convolution2D(128,
                                kernel_size,
                                padding='same',
                                input_shape=shape))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))
        model.add(Convolution2D(128,
                                kernel_size,
                                padding='same',
                                input_shape=shape))
        model.add(Dropout(0.1))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))

        model.add(Convolution2D(256,
                                kernel_size,
                                padding='same',
                                input_shape=shape))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))
        model.add(Convolution2D(256,
                                kernel_size,
                                padding='same',
                                input_shape=shape))
        model.add(Dropout(0.1))
        model.add(LeakyReLU(alpha_relu))

        model.add(Flatten())

        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        self.model = model
        
        adam_optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        
        
    def load_data(self, image_dir, gt_dir, training_size):
        files = os.listdir(image_dir)
        n = len(files)
        print("Loading " + str(n) + " images")
        imgs = [load_image(image_dir + files[i]) for i in range(n)]
        print(imgs[0][2])

        print("Loading " + str(n) + " images")
        gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
        print(files[0])

        self.X_train = imgs
        self.Y_train = gt_imgs
        
        

    def train(self, epochs=30, validation_split=0.2):
        
        # Step 0: Shuffle samples

        # Step 1: Split into validation and training set     
        
        # Step 2: Give weights to classes
        # FIXME correct?
        c_weight = {1: 2.8, 
                    0: 1.}
        
        # (depracated) Step 2 : other option, cut out data !
        # X, Y = get_equal_train_set_per_class(train_data, train_labels)

        # Step 3: Greate Generators
        generator = image_generator(self.X_train, self.Y_train, self.WINDOW_SIZE, batch_size = 32)

        
        # Step 4: Early stop and other Callbacks
        early_stop_callback = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=0, 
                                            mode='max', restore_best_weights=True)
        # Taken from Github model
        # FIXME does this work for accuracy on training set?
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                        verbose=1, mode='auto', min_delta=0, cooldown=0, min_lr=0) 
        
        # Save checkpoints
        filepath = "weights.{epoch:02d}-{acc:.2f}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1)
        # Finally, train the model !
        # Training
        self.model.fit_generator(generator,
                    steps_per_epoch=len(self.X_train * 16 * 16)/32, # FIXME how many steps per epoch?
                    epochs=epochs,
                    callbacks = [early_stop_callback, lr_callback],
                    class_weight=c_weight,
                    use_multiprocessing=True)
        
            
    def generate_images(self, prediction_training_dir, train_data_filename):
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        
        for i in range(1, constants.TRAINING_SIZE+1):
            pimg = get_prediction_with_mask(train_data_filename, i, self.model, self.WINDOW_SIZE)
            Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
            oimg = get_prediction_with_overlay(train_data_filename, i, self.model, self.WINDOW_SIZE)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        
        
    def generate_submission(self):
        createSubmission(self.model, self.WINDOW_SIZE)
  
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
        