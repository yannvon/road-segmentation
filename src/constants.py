""" Constants taken from the template file """

"""
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
RECORDING_STEP = 1000
"""

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
NUM_CHANNELS = 3 # RGB images
FOREGROUND_THRESHOLD = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
PIXEL_DEPTH = 255
NUM_LABELS = 2
