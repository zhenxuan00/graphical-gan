import tflib as lib

import numpy as np
import tensorflow as tf

def Conv3D(name, filter_len, input_dim, output_dim, filter_size, inputs, he_init=True, stride=1, stride_len=1, biases=True):
    """
    inputs: tensor of shape (N, L, H, W, C)

    returns: tensor of shape (N, L, H, W, C)
    """
    with tf.name_scope(name) as scope:
        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2 * filter_len
        fan_out = output_dim * filter_size**2 / (stride**2) * filter_len / stride_len

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        filter_values = uniform(
            filters_stdev,
            (filter_len, filter_size, filter_size, input_dim, output_dim)
        )

        filters = lib.param(name+'.Filters', filter_values)

        result = tf.nn.conv3d(
            input=inputs, 
            filter=filters, 
            strides=[1, stride_len, stride, stride, 1],
            padding='SAME',
            data_format='NDHWC'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros((1, 1, 1, 1, output_dim), dtype='float32')
            )

            result = tf.add(result, _biases)

        return result
