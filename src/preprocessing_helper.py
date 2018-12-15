import numpy as np
import os
import sys
import matplotlib.image as mpimg
import constants
import numpy


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
    padSize = (window_size-constants.IMG_PATCH_SIZE)//2
    padded = pad_image(im,padSize)
    for i in range(padSize,imgheight+padSize,constants.IMG_PATCH_SIZE):
        for j in range(padSize,imgwidth+padSize,constants.IMG_PATCH_SIZE):
            if is_2d:
                im_patch = padded[j-padSize:j+constants.IMG_PATCH_SIZE+padSize, i-padSize:i+constants.IMG_PATCH_SIZE+padSize]
            else:
                im_patch = padded[j-padSize:j+constants.IMG_PATCH_SIZE+padSize, i-padSize:i+constants.IMG_PATCH_SIZE+padSize, :]
            list_patches.append(im_patch)
    return list_patches

#pad an image 
def pad_image(img,padSize):
    is_2d = len(img.shape) < 3
    if is_2d:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize)),'reflect')
    else:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize),(0,0)),'reflect')
    
    

        