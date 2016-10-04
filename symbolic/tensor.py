from __future__ import print_function
from __future__ import division

class Tensor(object):
    """Represents a value produced by an operation."""

    def __init__(self, value=None, graph=None, op=None, name=None):
        self.value = value
        self.graph = graph
        self.op = op
        self.name = name

    def __add__(self, other):
        return self.graph.add(inputs=[self, other])

    def __sub__(self, other):
        return self.graph.sub(inputs=[self, other])

    def __mul__(self, other):
        return self.graph.mul(inputs=[self, other])

    def __truediv__(self, other):
        return self.graph.div(inputs=[self, other])
