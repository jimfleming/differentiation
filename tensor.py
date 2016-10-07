from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Tensor(object):
    """Tensor represents a value in a Graph."""

    def __init__(self, value, shape, op, graph, name):
        if shape is None:
            if value is not None and isinstance(value, np.ndarray):
                self.shape = value.shape
            else:
                self.shape = (1,)

        if name is None:
            self.name = 'Tensor'
        else:
            self.name = name

        self.value = value
        self.graph = graph
        self.op = op

    def __add__(self, other):
        return self.graph.add(self, other)

    def __sub__(self, other):
        return self.graph.sub(self, other)

    def __mul__(self, other):
        return self.graph.mul(self, other)

    def __div__(self, other):
        return self.graph.div(self, other)

    def __truediv__(self, other):
        return self.graph.div(self, other)

    def __neg__(self):
        return self.graph.neg(self)

    def __radd__(self, other):
        return self.graph.add(other, self)

    def __rsub__(self, other):
        return self.graph.sub(other, self)

    def __rmul__(self, other):
        return self.graph.mul(other, self)

    def __rdiv__(self, other):
        return self.graph.div(other, self)

    def __rtruediv__(self, other):
        return self.graph.div(other, self)

    def __repr__(self):
        return '{}("{}")'.format(type(self).__name__, self.name)
