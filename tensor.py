from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Tensor(object):
    """
    Tensor represents a value in a Graph. It includes a reference to the
    graph it belongs to and the op which produced the Tensor, if applicable.
    """

    def __init__(self, value, op, graph):
        self.value = value
        self.graph = graph
        self.op = op

    # overloads
    def __add__(self, other):
        return self.graph.add(self, other)

    def __sub__(self, other):
        return self.graph.sub(self, other)

    def __mul__(self, other):
        return self.graph.mul(self, other)

    def __truediv__(self, other):
        return self.graph.div(self, other)

    def __neg__(self):
        return self.graph.neg(self)

    # reverse overloads
    def __radd__(self, other):
        return self.graph.add(other, self)

    def __rsub__(self, other):
        return self.graph.sub(other, self)

    def __rmul__(self, other):
        return self.graph.mul(other, self)

    def __rtruediv__(self, other):
        return self.graph.div(other, self)
