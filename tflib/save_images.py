"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave
import imageio


def large_image(X, size=None):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    
    if size == None:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1

        nh, nw = rows, n_samples/rows
    else:
        nh, nw = size
        assert(nh * nw == n_samples)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    return img.astype('uint8')

def save_gifs(x, save_path, size=None):
    final_list = []
    for i in xrange(x.shape[1]):
        final_list.append(large_image(x[:,i,:,:,:], size=size))
    imageio.mimsave(save_path, final_list)

def save_images(X, save_path, size=None):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    
    if size == None:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1

        nh, nw = rows, n_samples/rows
    else:
        nh, nw = size
        assert(nh * nw == n_samples)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)