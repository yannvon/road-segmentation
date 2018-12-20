from constants import *
from dense_model import DenseModel
from basic_model import BasicModel
from deep_model import DeepModel
from preprocessing_helper import *
from postprocessing_helper import *

import os
import sys

# Set Train data Directory
data_dir = '../dataset/training/'

# Option 1: Load entire images
image_dir = data_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]

gt_dir = data_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]


# Choose model
model = DeepModel()
# model = BasicModel()
# model = DenseModel()
# model = TeslaModelS()

# Option 2: Model loads files itself
model.load_data(image_dir, gt_dir, constants.TRAINING_SIZE)

# Train model
model.train(epochs=200, validation_split=0.2)

# Create Training set images
# Set directory to save images
model.generate_images(imgs, gt_imgs)

# Predict Test Set -> Create submission and save Images
model.generate_submission()

# Save model
model.save()