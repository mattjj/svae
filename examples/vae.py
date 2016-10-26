from __future__ import division, print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import load_mnist, to_gpu
from svae.tf_nnet import init_mlp, tanh, gaussian_mean, make_loglike


expand = lambda x: tf.expand_dims(x, 1)

def kl(mu, sigmasq):
    return -0.5*tf.reduce_sum(1. + tf.log(sigmasq) - mu**2. - sigmasq)  # TODO consider numerical stability here!

def monte_carlo_elbo(encode, loglike, batch, eps):
    mu, sigmasq = encode(batch)
    z_sample = expand(mu) + expand(tf.sqrt(sigmasq)) * eps
    return loglike(z_sample, batch) - kl(mu, sigmasq)


if __name__ == '__main__':
    # settings
    latent_dim = 10   # number of latent dimensions
    obs_dim = 784     # dimensionality of observations

    num_samples = 1   # number of Monte Carlo samples per elbo evaluation
    step_size = 1e-3  # step size for the optimizer
    batch_size = 128  # number of examples in minibatch update
    num_epochs = 10   # number of passes over the training data

    # set up model and parameters
    encode, encode_params = init_mlp(obs_dim, [(200, tanh), (2*latent_dim, gaussian_mean)])
    decode, decode_params = init_mlp(latent_dim, [(200, tanh), (2*obs_dim, gaussian_mean)])
    loglike = make_loglike(decode)

    # load data and set up batch-getting function
    num_data, (train_images, train_labels, test_images, test_labels) = to_gpu(load_mnist())
    num_batches = num_data // batch_size

    def get_batch(step):
        start_index = (step % num_batches) * batch_size
        batch = lambda x: tf.slice(x, (start_index, 0), (batch_size, -1))
        return batch(train_images)

    # set up objective
    step = tf.Variable(0, trainable=False)
    eps = tf.random_normal((batch_size, num_samples, latent_dim))
    cost = -monte_carlo_elbo(encode, loglike, get_batch(step), eps)

    # set up ops
    train_op = tf.train.AdamOptimizer(step_size).minimize(cost, global_step=step)
    init_op = tf.initialize_all_variables()

    # run
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(num_epochs*num_batches):
            sess.run(train_op)
            if i % num_batches == 0:
                print(sess.run(cost))
