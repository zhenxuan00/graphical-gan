import tensorflow as tf

def l2(x, y):
    return tf.reduce_mean(tf.pow(x - y, 2))

def l1(x, y):
    return tf.reduce_mean(tf.abs(x - y))

def distance(x, y, d_type):
    xs = tf.shape(x)
    x = tf.reshape(x, [-1, xs[-1]])
    ys = tf.shape(y)
    y = tf.reshape(y, [-1, ys[-1]])
    if d_type is 'l1':
        return l1(x,y)
    elif d_type is 'l2':
        return l2(x,y)