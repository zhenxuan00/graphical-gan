import tensorflow as tf
import tflib as lib
import math

def kl_q_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std):
    q_z_var = tf.pow(q_z_std, 2)
    p_z_var = tf.pow(p_z_std, 2)
    mean_diff = tf.pow(p_z_mean - q_z_mean, 2)
    res_mat = .5*(tf.log(p_z_var/q_z_var) + (mean_diff + q_z_var) / p_z_var - 1)
    return tf.reduce_mean(tf.reduce_sum(res_mat, axis=1), axis=0)

def neg_log_likelihood_diagnoal_gaussian(x, mu, std):
    res_mat = .5*(tf.pow((x - mu)/std, 2) + tf.log(2*math.pi) + 2*tf.log(std))
    return tf.reduce_mean(tf.reduce_sum(res_mat, axis=1), axis=0)

def vae(real_x, p_x_mean, p_x_std, q_z_mean, q_z_std, p_z_mean, p_z_std, gen_params, lr=2e-4, beta1=.5):
    gen_cost = kl_q_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std)
    gen_cost += neg_log_likelihood_diagnoal_gaussian(real_x, p_x_mean, p_x_std)

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)

    return gen_cost, gen_train_op