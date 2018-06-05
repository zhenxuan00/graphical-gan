import tensorflow as tf
import tflib as lib

def score_function(f_k, p_k, c_v):
    # estimate the gradients of E_p(k|params) f(k)
    # the results is like (f(k) - cv) * Grad(log p(k|params))
    # or equivalently Grad(stop_gradient((f(k) - cv)) * log p(k|params)))

    return tf.stop_gradient(f_k - c_v) * tf.log(p_k)