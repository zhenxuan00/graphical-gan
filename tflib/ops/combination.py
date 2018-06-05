import tflib as lib

import numpy as np
import tensorflow as tf

def Ladder(inputs, input_dim, name):
    with tf.name_scope(name) as scope:
        zs = np.zeros(input_dim).astype('float32')
        os = np.ones(input_dim).astype('float32')

        a1 = lib.param(name + '.a1', zs)
        a2 = lib.param(name + '.a2', os)
        a3 = lib.param(name + '.a3', zs)
        a4 = lib.param(name + '.a4', zs)

        c1 = lib.param(name + '.c1', zs)
        c2 = lib.param(name + '.c2', os)
        c3 = lib.param(name + '.c3', zs)
        c4 = lib.param(name + '.c4', zs)

        b1 = lib.param(name + '.b1', zs)

        z_lat, u = inputs

        sigval = c1 + c2*z_lat
        sigval += c3*u + c4*z_lat*u
        sigval = tf.nn.sigmoid(sigval)
        z_est = a1 + a2 * z_lat + b1*sigval
        z_est += a3*u + a4*z_lat*u

        return z_est