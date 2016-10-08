from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np; np.random.seed(67)

from tqdm import trange

from graph import Graph
from session import Session

def main():
    # constants
    num_epochs = 10000

    # parameters
    graph = Graph()

    # data
    X = graph.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], name='X')
    y = graph.tensor([[0, 0, 0, 1]], name='y') # AND/MIN
    y = graph.tensor([[0, 1, 1, 1]], name='y') # OR/MAX

    W = graph.random_normal(shape=(3, 1), name='W')
    h = graph.sigmoid(graph.dot(X, W))

    loss_op = graph.mean(graph.square(graph.transpose(y) - h))

    W_grad, = graph.gradients(loss_op, [W])
    update_op = graph.assign_add(W, W_grad)

    sess = Session(graph)

    with trange(num_epochs) as pbar:
        for epoch in pbar:
            _, loss = sess.run([update_op, loss_op])
            pbar.set_description('{:.5f}'.format(loss))

if __name__ == '__main__':
    main()
