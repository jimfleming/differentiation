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
num_epochs = 100

# data (AND/MIN)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 0, 0, 1]])

# data (OR/MAX)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 1]])

# parameters
W = np.ones(shape=(3, 1))

with trange(num_epochs) as pbar:
    for epoch in pbar:
        # layers
        h = sigmoid(np.dot(X, W))

        # error
        loss = np.mean(np.abs(y.T - h))

        # gradient
        W_grad = np.dot(X.T, (y.T - h) * sigmoid_d(h))

        # update
        W += W_grad

        pbar.set_description('{:.5f}'.format(loss))
