from __future__ import division, print_function
import numpy as np
import tensorflow as tf
cross_entropy = tf.nn.softmax_cross_entropy_with_logits
from time import time

import data_mnist
from svae.tf_nnet import init_mlp, tanh, identity

# I wrote this example file to learn TensorFlow and develop a TensorFlow version
# of svae.nnet. It should serve as an example of how to use svae.nnet.


def negative_log_likelihood(mlp, batch):
    inputs, targets = batch
    return tf.reduce_mean(cross_entropy(mlp(inputs), targets))

def accuracy(mlp, inputs, targets):
    target_class = tf.argmax(targets, 1)
    predicted_class = tf.argmax(mlp(inputs), 1)
    return tf.reduce_mean(tf.to_float(tf.equal(target_class, predicted_class)))

def make_table(column_labels):
    lens = list(map(len, column_labels))
    print(' | '.join('{{:>{}}}'.format(l) for l in lens).format(*column_labels))
    row_format = ' | '.join(['{{:{}d}}'.format(lens[0])]
                            + ['{{:{}.4f}}'.format(l) for l in lens[1:]])
    def print_row(i, vals):
        print(row_format.format(i, *vals))
    return print_row

def load_mnist():
    # load as numpy arrays
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = data_mnist.mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N = train_images.shape[0]
    data = train_images, train_labels, test_images, test_labels

    # load onto gpu
    const = lambda x: tf.constant(x, tf.float32)
    with tf.device('/gpu:0'):
        tf_data = map(const, data)

    return N, tf_data

if __name__ == '__main__':
    # settings
    batch_size = 128
    step_size = 1e-3
    num_epochs = 100

    # set up model and parameters
    mlp = init_mlp(784, [(200, tanh), (100, tanh), (10, identity)])

    # load data and set up batch-getting function
    N, (train_images, train_labels, test_images, test_labels) = load_mnist()
    num_batches = N // batch_size

    def get_batch(step):
        start_index = (step % num_batches) * batch_size
        batch = lambda x: tf.slice(x, (start_index, 0), (batch_size, -1))
        return batch(train_images), batch(train_labels)

    # set up objective and other progress measures
    step = tf.Variable(0, trainable=False)
    cost = negative_log_likelihood(mlp, get_batch(step))
    train_accuracy = accuracy(mlp, train_images, train_labels)
    test_accuracy = accuracy(mlp, test_images, test_labels)

    # set up ops
    train_op = tf.train.AdamOptimizer(step_size).minimize(cost, global_step=step)
    init_op = tf.initialize_all_variables()

    # run
    with tf.Session() as sess:
        sess.run(init_op)

        print_row = make_table(['Epoch', 'Train accuracy', 'Test accuracy'])
        print_values = train_accuracy, test_accuracy

        for i in range(num_epochs*num_batches):
            sess.run(train_op)
            if i % num_batches == 0:
                print_row(i // num_batches, sess.run(print_values))
