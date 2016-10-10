from __future__ import print_function
from __future__ import division

import numpy as np

class Model(object):

    def __init__(self, graph):
        # placeholders
        self.X = graph.tensor(shape=(4, 2), name='X')
        self.y = graph.tensor(shape=(1, 4), name='y')

        # define our model parameters as tensors with weights initialized using He et al (sqrt(gain/fan_in)) and biases initialized to zeros

        # define layer 1 parameters
        W0 = graph.tensor(np.random.normal(scale=np.sqrt(1/2), size=(2, 4)).astype(np.float32), name='W0')
        b0 = graph.tensor(np.zeros(shape=(4,), dtype=np.float32))

        # define layer 2 parameters
        W1 = graph.tensor(np.random.normal(scale=np.sqrt(1/4), size=(4, 1)).astype(np.float32), name='W1')
        b1 = graph.tensor(np.zeros(shape=(1,), dtype=np.float32))

        # construct layer 1 output
        h0 = graph.sigmoid(graph.dot(self.X, W0) + b0, name='h0')

        # construct layer 2 output
        h1 = graph.sigmoid(graph.dot(h0, W1) + b1, name='h1')

        self.loss = graph.mean(graph.square(graph.transpose(self.y) - h1), name='loss')

        self.parameters = [W0, b0, W1, b1]
        gradients = graph.gradients(self.loss, self.parameters)
        self.update_op = graph.group([
            graph.assign_sub(param, grad) \
                for param, grad in zip(self.parameters, gradients)
        ])

    @property
    def size(self):
        return np.sum([np.prod(parameter.shape) for parameter in self.parameters])
