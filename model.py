from __future__ import print_function
from __future__ import division

import numpy as np

class Model(object):

    def __init__(self, learning_rate, batch_size, graph):
        # placeholders
        self.X = graph.tensor(shape=(batch_size, 784), name='X')
        self.y = graph.tensor(shape=(batch_size, 10), name='y')

        # define our model parameters as tensors with weights initialized using He et al (sqrt(gain/fan_in)) and biases initialized to zeros

        # define layer 1 parameters
        W0 = graph.tensor(np.random.normal(scale=np.sqrt(2/784), size=(784, 300)).astype(np.float32), name='W0')
        b0 = graph.tensor(np.zeros(shape=(300,), dtype=np.float32))

        # define layer 2 parameters
        W1 = graph.tensor(np.random.normal(scale=np.sqrt(1/300), size=(300, 10)).astype(np.float32), name='W1')
        b1 = graph.tensor(np.zeros(shape=(10,), dtype=np.float32))

        # construct layer 1 output
        h0 = graph.relu(graph.dot(self.X, W0) + b0, name='h0')

        # construct layer 2 output
        h1 = graph.softmax(graph.dot(h0, W1) + b1, name='h1')

        self.y_hat = h1

        epsilon = 1e-7

        inner_sum = self.y * graph.log(self.y_hat + epsilon)
        self.loss = graph.mean(-graph.sum(inner_sum, axis=1), name='loss')
        self.accuracy = graph.mean(graph.equal(graph.argmax(self.y, 1), graph.argmax(self.y_hat, 1)), name='accuracy')

        # TODO: fix bias add
        self.parameters = [W0, b0, W1, b1]
        self.gradients = graph.gradients(self.loss, self.parameters)
        self.update_op = graph.group([
            graph.assign_sub(param, learning_rate * grad) \
                for param, grad in zip(self.parameters, self.gradients)
        ])

    @property
    def size(self):
        return np.sum([np.prod(parameter.shape) for parameter in self.parameters])
