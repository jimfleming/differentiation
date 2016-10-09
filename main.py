from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np; np.random.seed(67)

from tqdm import trange

from graph import Graph
from session import Session

def main():
    # constants
    num_epochs = 100

    # parameters
    graph = Graph()

    # data
    X = graph.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], name='X')
    y = graph.tensor([[0, 0, 0, 1]], name='y') # AND/MIN
    y = graph.tensor([[0, 1, 1, 1]], name='y') # OR/MAX

    W_ = np.ones(shape=(3, 1))

    W = graph.tensor(shape=(3, 1), name='W')
    h = graph.sigmoid(graph.dot(X, W))

    delta = graph.transpose(y) - h
    delta_h = delta * (h * (1 - h))

    W_grad = graph.dot(graph.transpose(X), delta_h)

    sess = Session(graph)

    with trange(num_epochs) as pbar:
        for epoch in pbar:
            h_ = sess.run([h], feed_dict={W: W_})

            delta_ = sess.run([delta], feed_dict={W: W_})

            loss, = sess.run([delta], feed_dict={W: W_})
            loss = np.mean(np.abs(loss))

            delta_h_ = sess.run([delta_h], feed_dict={W: W_})

            # gradient
            W_grad_, = sess.run([W_grad], feed_dict={W: W_})

            # update
            W_ += W_grad_

            pbar.set_description('{:.5f}'.format(loss))

if __name__ == '__main__':
    main()
