from __future__ import division, print_function
import numpy as np
import tensorflow as tf

import data_mnist
from svae.tf_nnet import init_mlp, tanh, gaussian_mean, make_loglike


### vae functions

def kl(mu, sigmasq):
    return -0.5*tf.reduce_sum(1. + tf.log(sigmasq) - mu**2. - sigmasq)

def monte_carlo_elbo(encode, loglike, batch, eps):
    expand = lambda x: tf.expand_dims(x, 1)
    mu, sigmasq = map(expand, encode(batch))
    z_sample = mu + tf.sqrt(sigmasq) * eps
    return loglike(z_sample, batch) - kl(mu, sigmasq)

### data loading

def load_mnist():
    # load as numpy arrays
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = data_mnist.mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    num_data = train_images.shape[0]
    data = train_images, train_labels, test_images, test_labels

    # load onto gpu
    const = lambda x: tf.constant(x, tf.float32)
    with tf.device('/gpu:0'):
        tf_data = map(const, data)

    return num_data, tf_data


if __name__ == '__main__':
    # settings
    latent_dim = 10   # number of latent dimensions
    obs_dim = 784     # dimensionality of observations
    num_samples = 1   # number of Monte Carlo samples per elbo evaluation
    step_size = 1e-3  # step size for the optimizer

    # set up model and parameters
    encode, encode_params = init_mlp(obs_dim, [(200, tanh), (2*latent_dim, gaussian_mean)])
    decode, decode_params = init_mlp(latent_dim, [(200, tanh), (2*obs_dim, gaussian_mean)])
    loglike = make_loglike(decode)

    # load data and set up batch-getting function
    num_data, (train_images, train_labels, test_images, test_labels) = load_mnist()
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
