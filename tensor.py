from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Tensor(object):
    """
    Tensor represents a value in a Graph. It includes a shape, reference to the
    graph it belongs to and the op which produced the Tensor, if applicable.
    """

    def __init__(self, value, shape, op, graph, name):
        self.value = value

        if shape is None and value is None:
            raise ValueError('Must provide a value or shape to Tensor')

        if shape is None:
            if self.value is not None and isinstance(self.value, np.ndarray):
                self.shape = self.value.shape
            else:
                self.shape = ()
        else:
            self.shape = shape

        if name is None:
            self.name = 'Tensor'
        else:
            self.name = name

        self.graph = graph
        self.op = op

    # overloads
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

    def __pow__(self, other):
        return self.graph.power(self, other)

    def __neg__(self):
        return self.graph.neg(self)

    def __gt__(self, other):
        return self.graph.greater(self, other)

    # reverse overloads
    def __rpow__(self, other):
        return self.graph.power(other, self)

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

    # strings
    def __repr__(self):
        return '{}("{}", shape={})'.format(type(self).__name__, self.name, self.shape)
