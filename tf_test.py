from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tqdm import trange

def main():
    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y = tf.constant([[0, 1, 1, 0]], dtype=tf.float32)

    weights0 = tf.Variable(np.random.normal(size=(2, 4)), dtype=tf.float32)
    weights1 = tf.Variable(np.random.normal(size=(4, 1)), dtype=tf.float32)

    activations0 = tf.sigmoid(tf.matmul(X, weights0))
    activations1 = tf.sigmoid(tf.matmul(activations0, weights1))

    loss_op = tf.reduce_mean(tf.square(tf.transpose(y) - activations1))

    parameters = [weights0, weights1]
    gradients = tf.gradients(loss_op, parameters)

    update_op = tf.group(*[
        tf.assign(param, param - grad) \
            for param, grad in zip(parameters, gradients)
    ])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with trange(10000) as pbar_epoch:
            for _ in pbar_epoch:
                _, loss = sess.run([update_op, loss_op])
                pbar_epoch.set_description('loss: {:.8f}'.format(loss))

if __name__ == '__main__':
    main()
