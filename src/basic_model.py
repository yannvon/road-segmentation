from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import model_from_json
from sklearn.metrics import f1_score
import numpy as np


import constants
from preprocessing_helper import *
from postprocessing_helper import *


class BasicModel:
    """ A simple basic model following the provided template """
    
    WINDOW_SIZE = 16
    


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
        
        
        
    def load_data(self, train_data_filename, train_labels_filename, training_size):
        # Extract it into numpy arrays.
        self.train_data = extract_data(train_data_filename, training_size, self.WINDOW_SIZE)
        self.train_labels = extract_labels(train_labels_filename, training_size)
        
        

    def train(self, epochs=100, validation_split=0.1):
        
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
        # FIXME correct?
        c_weight = {1: 1., 
                    0: 3.}
        
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
        validation_generator = validation_datagen.flow(self.validation_data_split, self.validation_label_split, batch_size=32)

        
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
                    validation_steps=len(self.validation_data_split)/16)
        
    
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
        with open("basic_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("basic_model.h5")
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
