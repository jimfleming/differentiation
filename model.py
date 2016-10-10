from __future__ import print_function
from __future__ import division

import numpy as np

class Model(object):

    def __init__(self, g):
        # placeholders
        self.X = g.tensor(shape=(4, 2), name='X')
        self.y = g.tensor(shape=(1, 4), name='y')

        # layer 1
        W0 = g.tensor(np.random.normal(scale=np.sqrt(1/2), size=(2, 4)).astype(np.float32), name='W0')
        b0 = g.tensor(np.zeros(shape=(4,), dtype=np.float32))

        # layer 2
        W1 = g.tensor(np.random.normal(scale=np.sqrt(1/4), size=(4, 1)).astype(np.float32), name='W1')
        b1 = g.tensor(np.zeros(shape=(1,), dtype=np.float32))

        h0 = g.sigmoid(g.dot(self.X, W0) + b0, name='h0')
        h1 = g.sigmoid(g.dot(h0, W1) + b1, name='h1')

        self.loss = g.mean(g.square(g.transpose(self.y) - h1), name='loss')

        self.parameters = [W0, b0, W1, b1]
        gradients = g.gradients(self.loss, self.parameters)
        self.update_op = g.group([
            g.assign_sub(param, grad) \
                for param, grad in zip(self.parameters, gradients)
        ])

    @property
    def size(self):
        return np.sum([np.prod(parameter.shape) for parameter in self.parameters])
