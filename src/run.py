from constants import *
from dense_model import DenseModel
from basic_model import BasicModel
from deep_model import DeepModel
from preprocessing_helper import *
from postprocessing_helper import *

import os
import sys

# Step 1: choose whether to retrain the model or not
retrain = False
weights_to_load = "../trained_models/deep_model.h5"
# Set Train data set directory
data_dir = '../dataset/training/'

# Option 1: Load entire images
image_dir = data_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_dir = data_dir + "groundtruth/"
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]


# Step 2: Choose model
model = DeepModel()
# model = BasicModel()
# model = DenseModel()
# model = TeslaModelS()
# model = TopModel ;)

# Option 2: Model loads files itself
model.load_data(image_dir, gt_dir, constants.TRAINING_SIZE)

if retrain:
    # Train model
    model.train(epochs=200, validation_split=0.2)
else:
    model.load(weights_to_load)
    
# Create images displaying visualizations on prediction and test performance
# model.generate_images(imgs, gt_imgs)

# Predict Test Set -> Create submission and save Images and csv file
print("Generating submission, this can take a couple minutes..")
model.generate_submission()
print("Submission file was successfully generated.")

# Save model
if retrain:
    model.save()
    print("Trained model was successfully saved on disk.")