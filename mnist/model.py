from __future__ import print_function
from __future__ import division

import numpy as np

from utils import relu, relu_d, softmax, softmax_d, cross_entropy, label_accuracy, he_init, regularize_parameters, l2

class Model(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.regularization_strength = 1e-3
        self.learning_rate = 1e-3
        self.momentum = 0.9

        self.W0 = np.random.normal(scale=he_init(784), size=(784, 300)).astype(np.float32)
        self.b0 = np.zeros(shape=(300,), dtype=np.float32)

        self.W1 = np.random.normal(scale=he_init(300), size=(300, 10)).astype(np.float32)
        self.b1 = np.zeros(shape=(10,), dtype=np.float32)

        self.dW0_momentum = np.zeros_like(self.W0)
        self.db0_momentum = np.zeros_like(self.b0)
        self.dW1_momentum = np.zeros_like(self.W1)
        self.db1_momentum = np.zeros_like(self.b1)

    @property
    def parameters(self):
        return [self.W0, self.b0, self.W1, self.b1]

    @property
    def weights(self):
        return [self.W0, self.W1]

    @property
    def biases(self):
        return [self.b0, self.b1]

    @property
    def size(self):
        return np.sum([np.prod(parameter.shape) for parameter in self.parameters])

    def regularization(self):
        return (self.regularization_strength / self.batch_size) * regularize_parameters(self.weights, l2)

    def forward(self, X):
        h0 = relu(np.dot(X, self.W0) + self.b0)
        h1 = softmax(np.dot(h0, self.W1) + self.b1)
        return h1

    def backward(self, X, y):
        # forward
        h0 = relu(np.dot(X, self.W0) + self.b0)
        h1 = softmax(np.dot(h0, self.W1) + self.b1)

        # backward
        delta1 = h1 - y
        dW1 = np.dot(h0.T, delta1)
        db1 = np.sum(delta1, axis=0)

        delta0 = delta1.dot(self.W1.T) * relu_d(h0)
        dW0 = np.dot(X.T, delta0)
        db0 = np.sum(delta0, axis=0)

        # regularize
        dW1 += (self.regularization_strength / (2 * self.batch_size)) * self.W1
        dW0 += (self.regularization_strength / (2 * self.batch_size)) * self.W0

        dW0 = np.clip(dW0, -1, 1)
        db0 = np.clip(db0, -1, 1)
        dW1 = np.clip(dW1, -1, 1)
        db1 = np.clip(db1, -1, 1)

        # gradient descent
        self.dW0_momentum = self.momentum * self.dW0_momentum + self.learning_rate * dW0
        self.db0_momentum = self.momentum * self.db0_momentum + self.learning_rate * db0
        self.dW1_momentum = self.momentum * self.dW1_momentum + self.learning_rate * dW1
        self.db1_momentum = self.momentum * self.db1_momentum + self.learning_rate * db1

        self.W0 -= self.dW0_momentum
        self.b0 -= self.db0_momentum
        self.W1 -= self.dW1_momentum
        self.b1 -= self.db1_momentum
