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

# data (XOR)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 0]])

# parameters
W0 = np.random.normal(size=(3, 4))
W1 = np.random.normal(size=(4, 1))

with trange(num_epochs) as pbar:
    for epoch in pbar:
        # layer activations
        h0 = sigmoid(np.dot(X, W0))
        h1 = sigmoid(np.dot(h0, W1))

        h1_err = y.T - h1
        h1_delta = h1_err * sigmoid_d(h1)

        h0_err = np.dot(h1_delta, W1.T)
        h0_delta = h0_err * sigmoid_d(h0)

        W1_grad = np.dot(h0.T, h1_delta)
        W0_grad = np.dot(X.T, h0_delta)

        # update
        W1 += W1_grad
        W0 += W0_grad

        loss = np.mean(np.abs(h1_err))
        pbar.set_description('loss: {:.8f}'.format(loss))
