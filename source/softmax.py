# -*- coding: utf-8 -*-
#!/usr/bin/env python

""" Short description of this Python module.
Longer description of this module.
This program is for: .
"""

__authors__ = ["Qin Tian<qtian.whu@gmail.com>"]
__contact__ = "Qin Tian"
__copyright__ = "Copyright 2018"
__date__ = "18-1-22"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "Qin Tian"
__status__ = "test"
__version__ = "0.0.0"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

DATA_DIR = "../data"
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR,one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_maks = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_maks, tf.float32))

with tf.Session() as sess:
    # Train
    sess.run(tf.global_variables_initializer())

    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x:batch_xs, y_true:batch_ys})

    # Test
    ans = sess.run(accuracy, feed_dict={
        x: data.test.images,
        y_true: data.test.labels
    })

print("Accuracy: {:.4}%".format(ans*100))