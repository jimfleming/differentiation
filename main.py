from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np; np.random.seed(67)

from tqdm import trange, tqdm

from graph import Graph
from model import Model
from mnist import Dataset
from session import Session

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

def main():
    graph = Graph()

    X = graph.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    y = graph.tensor(np.array([[0, 0, 0, 1]])) # AND/MIN
    y = graph.tensor(np.array([[0, 1, 1, 1]])) # OR/MAX
    y = graph.tensor(np.array([[0, 1, 1, 0]])) # XOR

    W0 = graph.tensor(np.random.normal(size=(2, 4)))
    b0 = graph.tensor(np.random.normal(size=(4, 2)))

    W1 = graph.tensor(np.zeros(shape=(4,)))
    b1 = graph.tensor(np.zeros(shape=(2,)))

    h0 = graph.sigmoid(graph.dot(X, W0) + b0)
    h1 = graph.sigmoid(graph.dot(h0, W1) + b1)

    loss_op = graph.mean(graph.square(graph.transpose(y) - h))

    parameters = [W0, b0, W1, b1]
    gradients = graph.gradients(loss_op, parameters)
    update_op = graph.group([
        graph.assign_sub(param, grad) \
            for param, grad in zip(parameters, gradients)
    ])

    sess = Session(graph)
    with trange(NUM_EPOCHS) as pbar_epoch:
        for epoch in pbar_epoch:
            _, loss = sess.run([model.update_op, model.loss])
            pbar_epoch.set_description('loss: {:.8f}, accuracy: {:.2f}'.format(train_loss, train_accuracy))

if __name__ == '__main__':
    main()
