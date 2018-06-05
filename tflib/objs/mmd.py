import tensorflow as tf
import tflib as lib

def maximum_mean_discripancy(sample, data, batch_size, sigma=[2. , 5., 10., 20., 40., 80.]):
    x = tf.concat([sample, data], axis=0)
    xx = tf.matmul(x, x, transpose_b=True)
    x2 = tf.reduce_sum(tf.multiply(x, x), axis=1, keep_dims=True)
    exponent = tf.add(tf.add(xx, tf.scalar_mul(-.5, x2)), tf.scalar_mul(-.5, tf.transpose(x2)))

    s_samples = tf.scalar_mul(1./batch_size, tf.ones([tf.shape(sample)[0], 1]))
    s_data = tf.scalar_mul(-1./batch_size, tf.ones([tf.shape(data)[0], 1]))
    s_all = tf.concat([s_samples, s_data], axis=0)
    s_mat = tf.matmul(s_all, s_all, transpose_b=True)
    mmd_loss = 0.
    for s in sigma:
        kernel_val = tf.exp(tf.scalar_mul(1./s, exponent))
        mmd_loss += tf.reduce_sum(tf.multiply(s_mat, kernel_val))
    return tf.sqrt(mmd_loss)

def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2

def mix_rbf_mmd2(X, Y, sigmas=[2. , 5., 10., 20., 40., 80.], wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

def vegan_mmd(q_z, p_z, rec_penalty, gen_params, batch_size, lamb, lr=2e-4, beta1=.5):
    #mmd_cost = maximum_mean_discripancy(q_z, p_z, batch_size)
    gen_cost = lamb * mix_rbf_mmd2(q_z, p_z)
    gen_cost += rec_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)

    return gen_cost, gen_train_op