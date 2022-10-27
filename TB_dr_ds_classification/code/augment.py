import numpy as np
import scipy as sp
from numpy.random import permutation
from skimage.filters import gaussian
from global_constants import *


# Rotate image by a random angle between
# Input params:
#       data: image (array: H X W)
# Returns :
#       rotated input image
def rotate(data):
    angle = np.random.uniform(low=-MAX_ROTATION_DEGREES, high=MAX_ROTATION_DEGREES)
    return sp.ndimage.rotate(data, angle, reshape=False, mode='nearest')


# Shifts/translates input images by random pixels  in vertical and horizontal direction
# Input params:
#       data: image (array: H X W)
# Returns :
#       shifted input image
def translate(data):
    x = np.random.uniform(low=-MAX_TRANSLATION_PIXELS, high=MAX_TRANSLATION_PIXELS)
    y = np.random.uniform(low=-MAX_TRANSLATION_PIXELS, high=MAX_TRANSLATION_PIXELS)
    return sp.ndimage.shift(data, [x, y], mode='nearest')


# Shifts/translates input images by random pixels (-50 to 50) in vertical and horizontal direction
# Input params:
#       data: image (array: H X W)
# Returns:
#           image modified by either
#           a) applying gaussian noise with a sigma between 0 and 1.25
#           b) adjusting image intensities (by 75% to 125% original values for each pixel)
def random_changes(data):
    rnd = np.random.randint(low=0, high=2)
    if rnd == 0:
        rand = np.random.random(1)[0]
        aug_data = gaussian(data, sigma=NOISE_FACTOR * rand)
    else:
        rnd = np.random.uniform(low=1 - INTENSITY_CHANGE_PERCENT, high=1 + INTENSITY_CHANGE_PERCENT, size=data.shape)
        aug_data = data * rnd
    return aug_data

def augment(data, labels, size):
    """
    # Applies data transformations to input image and attaches to the end of the array
    Input params:
          data(np.ndarray): N X H X W, array of input images
          labels(np.ndarray): corresponding labels (N x l ), l can be 1 for 2 class or more for l classes
          size(int): augmentation factor
    returns:
          data(np.ndarray): all original and transformed images together
          labels(np.ndarray): all corresponding class labels
    """
    new_data = np.zeros((data.shape[0] * size, data.shape[1], data.shape[2]))
    new_labels = np.zeros((len(labels) * size))
    for k in range(size * len(labels)):
        cur_label = labels[k % len(labels)]
        d = data[k % len(labels)]
        d = rotate(d)
        d = translate(d)
        d = random_changes(d)
        new_data[k, :, :] = d
        new_labels[k] = cur_label
    new_data = np.concatenate((data, new_data), axis=0)
    new_labels = np.concatenate((labels, new_labels))
    perm = permutation(len(new_labels))
    data = new_data[perm, :, :]
    labels = new_labels[perm]
    return data, labels


