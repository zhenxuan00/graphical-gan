import tflib as lib

import numpy as np
import tensorflow as tf

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None


def MiniBatchLayer(name, num_inputs, num_kernels, dim_per_kernel, inputs):
    with tf.name_scope(name) as scope:
        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        weight_values = uniform(np.sqrt(2./num_inputs),(num_inputs, num_kernels, dim_per_kernel))

        weight = lib.param(
            name + '.W',
            weight_values
        )

        bias = lib.param(
            name + '.b',
            np.zeros((num_kernels,),dtype='float32')
        )

        activation = tf.tensordot(inputs, weight, [[1], [0]])
        abs_dif = (tf.reduce_sum(tf.abs(tf.expand_dims(activation, axis=-1) - tf.expand_dims(tf.transpose(activation, perm=[1, 2, 0]), axis=0)), axis=2)+ 1e6 * tf.expand_dims(tf.eye(tf.shape(inputs)[0]), axis=1))

        f = tf.reduce_sum(tf.exp(-abs_dif), axis=2)
        f += tf.expand_dims(bias, axis=0)
        return tf.concat([inputs, f], axis=1)