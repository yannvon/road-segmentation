import numpy as np
import os
import sys
import matplotlib.image as mpimg
import constants
import numpy
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import rotate


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def create_windows(im, window_size):
    list_patches = []
    is_2d = len(im.shape) < 3
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    padSize = (window_size - constants.IMG_PATCH_SIZE)//2
    padded = pad_image(im, padSize)
    for i in range(padSize, imgheight + padSize, constants.IMG_PATCH_SIZE):
        for j in range(padSize,imgwidth + padSize, constants.IMG_PATCH_SIZE):
            if is_2d:
                im_patch = padded[j-padSize:j+constants.IMG_PATCH_SIZE+padSize, i-padSize:i+constants.IMG_PATCH_SIZE+padSize]
            else:
                im_patch = padded[j-padSize:j+constants.IMG_PATCH_SIZE+padSize, i-padSize:i+constants.IMG_PATCH_SIZE+padSize, :]
            list_patches.append(im_patch)
    return list_patches

#pad an image 
def pad_image(img, padSize):
    is_2d = len(img.shape) < 3
    if is_2d:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize)),'reflect')
    else:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize),(0,0)),'reflect')
    
    
#source for function rot(image, xy, angle)
#https://stackoverflow.com/questions/46657423/rotated-image-coordinates-after-scipy-ndimage-interpolation-rotate
def rot(image, xy, angle):
    im_rot = rotate(image, angle, reshape=False) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot

def image_generator(images, ground_truths, window_size, batch_size = 64):
    np.random.seed(0)
    imgWidth = images[0].shape[0]
    imgHeight = images[0].shape[1]
    half_patch = constants.IMG_PATCH_SIZE // 2
    
    padSize = (window_size - constants.IMG_PATCH_SIZE) // 2
    paddedImages = []
    for image in images:
        paddedImages.append(pad_image(image,padSize))
    
    rotatedImages = []
    rotatedGroundTruths = []
    
    # We will augment the dataset x8, by including every 45 degree rotations
    # for rotations not multiple of 90 degree, the image needs to be interpolated, hence we limit the position
    # of the window to be inside the original image. We can visualize the authorised positions as a square of
    # side length 1/sqrt(2) * the original side length of the image.
    rotation_thresh = 4 * len(images)
    rotations = [0, 90, 180, 270, 45, 135, 225, 315]
    for i in range(len(images)):
        for j in range(8):
            rotatedImages.append(rot(paddedImages[i], np.array([imgWidth, imgHeight]), rotations[j]))
            rotatedGroundTruths.append(rot(ground_truths[i], np.array([imgWidth, imgHeight]), rotations[j]))
    
    while True:
        batch_input = []
        batch_output = [] 
                
        for i in range(batch_size):
            x = np.empty((window_size, window_size, 3))
            y = np.empty((window_size, window_size, 3))
            
            randomIndex = np.random.randint(0, len(rotatedImages))  
            img = rotatedImages[randomIndex]
            gt = rotatedGroundTruths[randomIndex]
            
            # we need to limit possible centers to avoid having a window in an interpolated part of the image
            # we limit ourselves to a square of width 1/sqrt(2) smaller
            if(randomIndex > rotation_thresh):
                boundary = int((imgWidth - imgWidth / np.sqrt(2)) / 2)
            else:
                boundary = 0
                
            center_x = np.random.randint(half_patch + boundary, imgWidth  - half_patch - boundary)
            center_y = np.random.randint(half_patch + boundary, imgHeight - half_patch - boundary)
            
            x = img[center_x - half_patch:center_x + half_patch + 2 * padSize,
                    center_y  - half_patch:center_y + half_patch + 2 * padSize]
            y = gt[center_x - half_patch : center_x + half_patch,
                   center_y - half_patch : center_y + half_patch]
            
            # vertical
            if(np.random.randint(0, 2)):
                x = np.flipud(x)
            
            # horizontal
            if(np.random.randint(0, 2)):
                x = np.fliplr(x)
            
            label = [0., 1.] if (np.array([np.mean(y)]) > constants.FOREGROUND_THRESHOLD) else [1., 0.]
            
            batch_input.append(x)
            batch_output.append(label)


        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )

        yield( batch_x, batch_y )        
        