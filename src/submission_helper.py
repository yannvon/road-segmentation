import os
import numpy
import matplotlib.image as mpimg
import re
from PIL import Image


import constants
from helper import load_image


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
            100.0 *
            numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
            predictions.shape[0])


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * constants.PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img * constants.PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# Get prediction for given input image
def get_prediction(img,model):
    data = numpy.asarray(img_crop(img, constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE))
    output = model.predict(data)
    output_prediction = output
    img_prediction = label_to_img(img.shape[0], img.shape[1], constants.IMG_PATCH_SIZE, constants.IMG_PATCH_SIZE, output_prediction)
    return img_prediction

# Get a concatenation of the prediction and image for given input file
def get_prediction_with_mask(filename, image_idx, model):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img, model)
    cimg = concatenate_images(img, img_prediction)
    print(cimg.shape)
    return cimg

# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, model):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img, model)
    oimg = make_img_overlay(img, img_prediction)

    return oimg

# assign a label to a patch
def patch_to_label(patch):
    df = numpy.mean(patch)
    if df > constants.FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


def mask_to_submission_strings(img_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", img_filename).group(0))
    im = mpimg.imread(img_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

        # load test image test them and return submission


def createSubmission(model):
    submission_filename = 'dummy_submission.csv'
    image_filenames = []
    prediction_test_dir = "predictions_test/"
    pred_filenames = []
    for i in range(1, 51):
        image_filename = '../dataset/test_set_images/test_' + str(i) + "/test_" + str(i) + ".png"
        image_filenames.append(image_filename)
    test_imgs = [load_image(image_filenames[i]) for i in range(50)]
    for i in range(50):
        pimg = get_prediction(test_imgs[i], model)
        w = pimg.shape[0]
        h = pimg.shape[1]
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(pimg)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        pred_filename = prediction_test_dir + "prediction_" + str(i + 1) + ".png"
        Image.fromarray(gt_img_3c).save(pred_filename)
        pred_filenames.append(pred_filename)
    masks_to_submission(submission_filename, *pred_filenames)