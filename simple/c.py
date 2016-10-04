from __future__ import print_function
from __future__ import division

import numpy as np
np.random.seed(67)

from tqdm import trange

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def l2(W):
    return np.sum(np.square(W))

def cross_entropy(y_preds, y_labels):
    return -np.mean(np.sum(y_labels * np.log(y_preds), axis=1))

def mean_squared_error(y_preds, y_labels):
    return np.mean(np.sum(np.square(y_preds - y_labels), axis=1))

# constants
num_epochs = 60000
learning_rate = 1e-2
regularization_strength = 1e-5

# data (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# parameters
W0 = np.random.normal(size=(2, 4))
W1 = np.random.normal(size=(4, 2))
b0 = np.zeros(shape=(4,))
b1 = np.zeros(shape=(2,))

losses = []
with trange(num_epochs) as pbar:
    for epoch in pbar:
        # forward prop
        h0 = np.tanh(np.dot(X, W0) + b0)
        h1 = softmax(np.dot(h0, W1) + b1)

        # backprop
        delta1 = h1 - y
        dW1 = np.dot(h0.T, delta1)
        db1 = np.sum(delta1, axis=0)

        delta0 = delta1.dot(W1.T) * (1 - np.square(h0))
        dW0 = np.dot(X.T, delta0)
        db0 = np.sum(delta0, axis=0)
        
        dW1 += regularization_strength * 2 * W1
        dW0 += regularization_strength * 2 * W0
 
        # gradient descent
        W0 += -learning_rate * dW0
        b0 += -learning_rate * db0

        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1

        # loss
        loss = cross_entropy(h1, y)
        loss += regularization_strength * np.sum([l2(W0), l2(W1)])
        losses.append(loss)

        pbar.set_description('loss: {:.8f}'.format(loss))
