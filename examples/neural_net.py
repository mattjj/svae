from __future__ import division, print_function
import numpy as np
import tensorflow as tf
cross_entropy = tf.nn.softmax_cross_entropy_with_logits
from time import time

from data import load_mnist, to_gpu
from svae.tf_nnet import init_mlp, tanh, identity

# I wrote this example file to learn TensorFlow and develop a TensorFlow version
# of svae.nnet. It should serve as an example of how to use svae.nnet.


### neural net functions

def negative_log_likelihood(mlp, batch):
    inputs, targets = batch
    return tf.reduce_mean(cross_entropy(mlp(inputs), targets))

def accuracy(mlp, inputs, targets):
    target_class = tf.argmax(targets, 1)
    predicted_class = tf.argmax(mlp(inputs), 1)
    return tf.reduce_mean(tf.to_float(tf.equal(target_class, predicted_class)))

### printing

def make_table(column_labels):
    column_labels = [' Epoch', '  Time'] + column_labels
    lens = list(map(len, column_labels))
    print(' | '.join('{{:>{}}}'.format(l) for l in lens).format(*column_labels))

    time_format = '{{:{}.2f}}'
    epoch_format = '{{:{}d}}'
    format_strings = [epoch_format, time_format] + ['{{:{}.4f}}'] * len(lens[2:])
    row_format = ' | '.join(s.format(l) for s, l in zip(format_strings, lens))

    outer = {'i': 0, 'start_time': time()}  # no nonlocal keyword in Python 2.7
    def print_row(vals):
        elapsed_time = time() - outer['start_time']
        print(row_format.format(outer['i'], elapsed_time, *vals))
        outer['i'] += 1

    return print_row

if __name__ == '__main__':
    # settings
    batch_size = 128
    step_size = 1e-3
    num_epochs = 100

    # set up model and parameters
    mlp, mlp_params = init_mlp(784, [(200, tanh), (100, tanh), (10, identity)])

    # load data and set up batch-getting function
    num_data, (train_images, train_labels, test_images, test_labels) = to_gpu(load_mnist())
    num_batches = num_data // batch_size

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

        print_row = make_table(['Train accuracy', 'Test accuracy'])
        print_values = train_accuracy, test_accuracy

        for i in range(num_epochs*num_batches):
            sess.run(train_op)
            if i % num_batches == 0:
                print_row(sess.run(print_values))
