import numpy as np

import scipy.misc
import os
import urllib
import gzip
import cPickle as pickle
from glob import glob
from scipy.misc import imsave, imresize

def celeba_generator(batch_size, images):
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)

        for i in xrange(len(images) / batch_size):
            yield images[i*batch_size:(i+1)*batch_size]

    return get_epoch

def load(batch_size, data_dir, num_dev=5000):
    data = np.load(os.path.join(data_dir, 'celebA_64x64.npy'))
    
    data = data.reshape(data.shape[0], -1)

    rng_state = np.random.get_state()
    np.random.shuffle(data)

    x_train = data[num_dev:]
    x_test = data[:num_dev]
    
    return (
        celeba_generator(batch_size, x_train), 
        celeba_generator(batch_size, x_test)
    )

def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def center_crop(x, resize_h=64, resize_w=64):
    h, w = x.shape[:2]
    assert(h >= w)
    new_h = int(h * resize_w / w)
    x = imresize(x, (new_h, resize_w))
    margin = int(round((new_h - resize_h)/2))
    return x[margin:margin+resize_h]

def transform(image, resize_height=64, resize_width=64):
    cropped_image = center_crop(image, resize_height, resize_width)
    return np.array(cropped_image)

def get_image(image_path, resize_height=64, resize_width=64, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, resize_height, resize_width)

def print_array(x):
    print x.shape, x.dtype, x.max(), x.min()

def convert_to_numpy(data_path, size=64):
    data = glob(os.path.join(data_path, '*.jpg'))
    
    sample_files = data[0:202599]
    sample = [get_image(sample_file,
                resize_height=size,
                resize_width=size,
                grayscale=False) for sample_file in sample_files]
    sample_inputs = np.array(sample)
    sample_inputs = np.transpose(sample_inputs, [0, 3, 1, 2])
    print_array(sample_inputs)
    np.save('celebA_64x64.npy', sample_inputs)