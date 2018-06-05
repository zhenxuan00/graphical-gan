import numpy as np

import os
import urllib
import gzip
import cPickle as pickle


def GetRandomTrajectory(step_length, seq_length, batch_size, image_size,digit_size):
    canvas_size = image_size - digit_size
    
    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((seq_length, batch_size))
    start_x = np.zeros((seq_length, batch_size))
    for i in xrange(seq_length):
        # Take a step along velocity.
        y += v_y * step_length
        x += v_x * step_length

        # Bounce off edges.
        for j in xrange(batch_size):
            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
            if x[j] >= 1.0:
                x[j] = 1.0
                v_x[j] = -v_x[j]
            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
            if y[j] >= 1.0:
                y[j] = 1.0
                v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

def Overlap(a, b):
    return np.maximum(a, b)
    #return b

def moving_mnist_generator_video(data_all, seq_length, batch_size):
    images, labels = data_all
    images = images.reshape([-1, 28, 28])
    image_size = 64
    num_digits = 1
    step_length = 0.1
    digit_size = 28
    
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        start_y, start_x = GetRandomTrajectory(step_length = step_length, seq_length = seq_length, batch_size = images.shape[0]*num_digits, image_size = image_size, digit_size = digit_size)

        data = np.zeros((images.shape[0], seq_length, image_size, image_size), dtype=np.float32)
        
        for j in xrange(images.shape[0]):
            for n in xrange(num_digits):

                digit_image = images[j, :, :]
                
                # generate video
                for i in xrange(seq_length):
                    top    = start_y[i, j * num_digits + n]
                    left   = start_x[i, j * num_digits + n]
                    bottom = top  + digit_size
                    right  = left + digit_size

                    data[j, i, top:bottom, left:right] = Overlap(data[j, i, top:bottom, left:right], digit_image)
        
        data = data.reshape(images.shape[0], seq_length, image_size*image_size)

        for ind in xrange(data.shape[0]/ batch_size):
             yield data[ind*batch_size:(ind+1)*batch_size], labels[ind*batch_size:(ind+1)*batch_size]

    return get_epoch

def load_video(seq_length, batch_size, cla=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)
    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)
    train_all_x = np.concatenate([train_data[0], dev_data[0]], axis=0)
    train_all_y = np.concatenate([train_data[1], dev_data[1]], axis=0)

    if cla is not None:
        train_all_x = train_all_x[train_all_y == cla]
        train_all_y = train_all_y[train_all_y == cla]
        test_x, test_y = test_data
        test_x = test_x[test_y == cla]
        test_y = test_y[test_y == cla]
        test_data = (test_x, test_y)

    return (moving_mnist_generator_video((train_all_x, train_all_y), seq_length, batch_size), moving_mnist_generator_video(test_data, seq_length, batch_size))

def moving_mnist_generator_image(image, seq_length, batch_size):
    assert batch_size % seq_length == 0
    video_gen = moving_mnist_generator_video(image, seq_length, batch_size/seq_length)
    data = []
    label = []
    for v, y in video_gen():
        data.append(v.reshape([batch_size, 64*64]))
        label.append(np.tile(y.reshape(-1, 1), [1, seq_length]).reshape(-1))
    data = np.vstack(data)
    label = np.concatenate(label, axis=0)
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(rng_state)
        np.random.shuffle(label)

        for i in xrange(len(data) / batch_size):
            yield data[i*batch_size:(i+1)*batch_size], label[i*batch_size:(i+1)*batch_size]
    return get_epoch

def load_image(seq_length, batch_size, cla=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    if not os.path.isfile(filepath):
        print "Couldn't find MNIST dataset in /tmp, downloading..."
        urllib.urlretrieve(url, filepath)
    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)
    train_all_x = np.concatenate([train_data[0], dev_data[0]], axis=0)
    train_all_y = np.concatenate([train_data[1], dev_data[1]], axis=0)

    if cla is not None:
        train_all_x = train_all_x[train_all_y == cla]
        train_all_y = train_all_y[train_all_y == cla]
        test_x, test_y = test_data
        test_x = test_x[test_y == cla]
        test_y = test_y[test_y == cla]
        test_data = (test_x, test_y)

    return (moving_mnist_generator_image((train_all_x, train_all_y), seq_length, batch_size), moving_mnist_generator_image(test_data, seq_length, batch_size))