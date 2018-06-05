import numpy as np

import os
import urllib
import gzip
import cPickle as pickle
from scipy.io import loadmat

def maybe_download(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', data_dir+'/train_32x32.mat', _progress)
        filepath, _ = urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', data_dir+'/test_32x32.mat', _progress)

def svhn_generator(data, batch_size):
    images, labels = data

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch

def load(batch_size, data_dir):
    maybe_download(data_dir)
    train_data = loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    trainx = train_data['X']
    trainy = train_data['y'].flatten()
    trainy[trainy==10] = 0
    test_data = loadmat(os.path.join(data_dir, 'test_32x32.mat'))
    testx = test_data['X']
    testy = test_data['y'].flatten()
    testy[testy==10] = 0
    trainx = np.transpose(trainx, [3, 2, 0, 1])
    testx = np.transpose(testx, [3, 2, 0, 1])
    trainx = trainx.reshape([-1, 32*32*3])
    testx = testx.reshape([-1, 32*32*3])
    return (
        svhn_generator((trainx, trainy), batch_size), 
        svhn_generator((testx, testy), batch_size)
    )