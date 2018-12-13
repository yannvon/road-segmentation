import numpy as np

def get_equal_train_set_per_class(train_data, train_labels):
    """ In charge of making sure training data is equally split between both classes """

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]

    print(train_labels)
    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
    return train_data, train_labels


# pad an image (with 3RGB dim with) of padSize by mirroring border
def pad_image(img, padSize):
    return np.lib.pad(img, ((padSize, padSize), (padSize, padSize), (0, 0)), 'reflect')


# pad an ground_truth image
def pad_gt_img(img, padSize):
    return np.lib.pad(img, ((padSize, padSize), (padSize, padSize)), 'reflect')


# plt.imshow((pad_image(imgs[0], 100)), cmap='Greys_r')