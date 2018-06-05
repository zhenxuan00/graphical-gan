import numpy as np
import tensorflow as tf
import tflib as lib
import math

def mixture_gaussian(n_samples, n_coms, dim_z, mu, std):
    # prior
    pi = tf.constant(np.ones(n_coms).astype('float32')/n_coms)
    dist = tf.distributions.Categorical(probs=pi)
    # sample components
    k = tf.cast(tf.one_hot(indices=dist.sample(n_samples), depth=n_coms), tf.float32)
    # sample noise and transfer
    mu_k = tf.matmul(k, mu)
    std_k = tf.matmul(k, std)
    eps = tf.random_normal([n_samples, dim_z])
    return tf.add(mu_k, tf.multiply(std_k, eps))

def log_likelihood_diagnoal_gaussian(x, mu, std):
    res_mat = -.5*(tf.pow((x - mu)/std, 2) + tf.log(2*math.pi) + 2*tf.log(std))
    return tf.reduce_sum(res_mat, axis=-1)

def log_likelihood_mixture_gaussian(x, mu, std):
    x = tf.expand_dims(x, axis=1)
    mu = tf.expand_dims(mu, axis=0)
    std = tf.expand_dims(std, axis=0)
    res_mat = log_likelihood_diagnoal_gaussian(x, mu, std) 
    # log sum exp trick to aovid overflow
    res_max = tf.reduce_max(res_mat, axis=1)
    res_max_keep = tf.expand_dims(res_max, axis=1)
    return tf.log(tf.reduce_mean(tf.exp(res_mat - res_max_keep), axis=1)) + res_max

def log_likelihood_mixture_mixture_gaussian(x, mu_q, std_q, mu_p, std_p, n_coms):
    x_q = tf.expand_dims(x, axis=1) # nz x 1 x dz
    mu_q = tf.expand_dims(mu_q, axis=0) # 1 x nx x dz
    std_q = tf.expand_dims(std_q, axis=0)
    res_mat_1 = log_likelihood_diagnoal_gaussian(x_q, mu_q, std_q) # nz x nx
    res_mat_2 = log_likelihood_diagnoal_gaussian(x, mu_p, std_p) # nz 
    res_mat_2 = tf.tile(tf.expand_dims(res_mat_2, axis=1), [1, n_coms])
    res_mat  = tf.concat([res_mat_1, res_mat_2], axis=1)

    # log sum exp trick to aovid overflow
    res_max = tf.reduce_max(res_mat, axis=1)
    res_max_keep = tf.expand_dims(res_max, axis=1)
    return tf.log(tf.reduce_mean(tf.exp(res_mat - res_max_keep), axis=1)) + res_max

def kl_q_aggregated_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std, n_samples, n_coms, dim_z):
    # sample z from q(z)
    z = mixture_gaussian(n_samples, n_coms, dim_z, q_z_mean, q_z_std)
    log_q = log_likelihood_mixture_gaussian(z, q_z_mean, q_z_std)
    log_p = log_likelihood_diagnoal_gaussian(z, p_z_mean, p_z_std)
    return tf.reduce_mean(log_q - log_p, axis=0)

def ikl_q_aggregated_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std, n_samples, dim_z):
    # sample z from p(z)
    z = tf.random_normal([n_samples, dim_z])
    log_q = log_likelihood_mixture_gaussian(z, q_z_mean, q_z_std)
    log_p = log_likelihood_diagnoal_gaussian(z, p_z_mean, p_z_std)
    return tf.reduce_mean(log_p - log_q, axis=0)

def jsd_q_aggregated_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std, n_samples, n_coms, dim_z):
    # sample z from q(z)
    z_1 = mixture_gaussian(n_samples, n_coms, dim_z, q_z_mean, q_z_std)
    log_q = log_likelihood_mixture_gaussian(z_1, q_z_mean, q_z_std)
    log_m_1 = log_likelihood_mixture_mixture_gaussian(z_1, q_z_mean, q_z_std, p_z_mean, p_z_std, n_coms)

    # sample z from p(z)
    z_2 = tf.random_normal([n_samples, dim_z])
    log_p = log_likelihood_diagnoal_gaussian(z_2, p_z_mean, p_z_std)
    log_m_2 = log_likelihood_mixture_mixture_gaussian(z_2, q_z_mean, q_z_std, p_z_mean, p_z_std, n_coms)
    return tf.reduce_mean(.5*(log_q - log_m_1 + log_p -log_m_2), axis=0)

def vegan_jsd(q_z_mean, q_z_std, p_z_mean, p_z_std, rec_penalty, gen_params, z_samples, batchsize, dim_z, lamb, lr=2e-4, beta1=.5):
    gen_cost = lamb * jsd_q_aggregated_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std, z_samples, batchsize, dim_z)
    gen_cost += rec_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)

    return gen_cost, gen_train_op

def vegan_kl(q_z_mean, q_z_std, p_z_mean, p_z_std, rec_penalty, gen_params, z_samples, batchsize, dim_z, lamb, lr=2e-4, beta1=.5):
    gen_cost = lamb * kl_q_aggregated_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std, z_samples, batchsize, dim_z)
    gen_cost += rec_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)

    return gen_cost, gen_train_op

def vegan_ikl(q_z_mean, q_z_std, p_z_mean, p_z_std, rec_penalty, gen_params, z_samples, dim_z, lamb, lr=2e-4, beta1=.5):
    gen_cost = lamb * ikl_q_aggregated_p_diagonal_gaussian(q_z_mean, q_z_std, p_z_mean, p_z_std, z_samples, dim_z)
    gen_cost += rec_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=lr, 
        beta1=beta1
    ).minimize(gen_cost, var_list=gen_params)

    return gen_cost, gen_train_op