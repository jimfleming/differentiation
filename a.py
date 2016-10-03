from __future__ import print_function
from __future__ import division

import numpy as np
np.random.seed(67)

from tqdm import trange

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    return x * (1 - x)

# constants
num_epochs = 10000

# data (AND/MIN)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 0, 0, 1]])

# data (OR/MAX)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 1]])

# parameters
W = np.random.normal(size=(3, 1))

with trange(num_epochs) as pbar:
    for epoch in pbar:
        # layer activations
        h = sigmoid(np.dot(X, W))

        # error
        err = y.T - h
        delta = err * sigmoid_d(h)

        # gradient
        W_grad = np.dot(X.T, delta)

        # update
        W += W_grad

        loss = np.mean(np.abs(err))
        pbar.set_description('{:.5f}'.format(loss))
