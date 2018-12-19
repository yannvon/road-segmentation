from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from keras.models import model_from_json
from sklearn.metrics import f1_score
import numpy as np

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.regularizers import l2

# FIXME too many imports, remove unnecessary
from keras import callbacks
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam


import constants
from preprocessing_helper import *
from postprocessing_helper import *


class NewModel:
    """ A simple model inspired by the VGG model """
    
    WINDOW_SIZE = 100
    OUTPUT_FILENAME = "vgg_model_wind" + str(WINDOW_SIZE)

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
        
        adam_optimizer = Adam(lr=0.0005)
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
        
        # FIXME this will introduce randomness, i.e. different print on kaggle and aws !
        

    def train(self, epochs=30, validation_split=0.2):
        
       # Step 0: Shuffle samples
        np.random.seed(0)
        np.random.shuffle(self.X_train)
        # resetting the seed allows for an identical shuffling between y and x
        np.random.seed(0)
        np.random.shuffle(self.Y_train)

        # Step 1: Split into validation and training set
        split_index = int(len(self.X_train) * (1 - validation_split))
        self.train_data_split = self.X_train[0:split_index]
        self.validation_data_split = self.X_train[split_index:len(self.X_train)]
        self.train_label_split = self.Y_train[0:split_index]
        self.validation_label_split = self.Y_train[split_index:len(self.Y_train)]    
        
        # Step 2: Solve underrepresentation problem
        # Option 1: Give weights to classes
        # c_weight = {1: 2.8, 
        #            0: 1.}
        
        # Option 2: downsample data
        # X, Y = get_equal_train_set_per_class(train_data, train_labels)
        
        # Option 3: upsample data 
        # This is done here, in the image_generator directly !
        
        # Note: Should we also upsample validation set?
        
        # Step 3: Greate Generators
        train_generator = image_generator(self.train_data_split,
                                          self.train_label_split,
                                          self.WINDOW_SIZE,
                                          batch_size = 32, 
                                          upsample=True)
        
        validation_generator = image_generator(self.validation_data_split,
                                               self.validation_label_split,
                                               self.WINDOW_SIZE,
                                               batch_size = 32,
                                               upsample=True)

        # Step 4: Early stop and other Callbacks
        early_stop_callback = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=1, 
                                            mode='auto', restore_best_weights=True)
        # Taken from Github model
        # FIXME does this work for accuracy on training set?
        lr_callback = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,
                                        verbose=1, mode='auto', min_delta=0, cooldown=0, min_lr=0) 
        
        # Save checkpoints
        filepath = "weights.{epoch:02d}-{acc:.2f}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='acc', verbose=0, save_best_only=True,
                                            save_weights_only=False, mode='auto', period=1)
        
        # Save data for TensorBoard
        # Check here for more info: https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L669
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, 
                                          write_graph=True, write_grads=True,
                                          write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                          embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

        #Log each epoch in a csv file
        csv_logger = CSVLogger(self.OUTPUT_FILENAME + '_training.log')

        # Finally, train the model !
        # Training  
        self.model.fit_generator(train_generator,
                    validation_data = validation_generator,
                    steps_per_epoch=len(self.X_train * 16 * 16) / 32, # FIXME how many steps per epoch?
                    epochs=epochs,
                    callbacks = [early_stop_callback, lr_callback, checkpoint_callback, tensorboard_callback, csv_logger],
                    use_multiprocessing=True,
                    validation_steps=len(self.validation_data_split) * 16 * 16 * 2 / 32)
        
            
    def generate_images(self, imgs, gt_imgs):
        prediction_training_dir = "predictions_training/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        
        for i in range(1, constants.TRAINING_SIZE+1):
            pimg = get_prediction_with_mask(imgs[i-1], self.model, self.WINDOW_SIZE)
            Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
            oimg = get_prediction_with_overlay(imgs[i-1], self.model, self.WINDOW_SIZE)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        
        checkImageTrainSet(self.model, imgs, gt_imgs, self.WINDOW_SIZE)
        
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
        # FIXME
        ##load json and create model
        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #loaded_model = model_from_json(loaded_model_json)
        
        ##load weights into new model
        self.model.load_weights(self.OUTPUT_FILENAME + ".h5")
        #loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1])
        print("Loaded model from disk")
        