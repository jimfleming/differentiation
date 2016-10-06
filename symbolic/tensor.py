from __future__ import print_function
from __future__ import division

class Tensor(object):
    """Represents a value produced by an operation."""

    def __init__(self, value=None, graph=None, op=None, name='Tensor'):
        self.value = value
        self.graph = graph
        self.op = op
        self.name = name

    def __add__(self, other):
        return self.graph.add(self, other)

    def __radd__(self, other):
        return self.graph.add(other, self)

    def __sub__(self, other):
        return self.graph.sub(self, other)

    def __mul__(self, other):
        return self.graph.mul(self, other)

    def __truediv__(self, other):
        return self.graph.div(self, other)

    def __str__(self):
        return '{}("{}")'.format(type(self).__name__, self.name)

    def __repr__(self):
        return str(self)

    def eval(self, context):
        if self.op:
            return self.op.compute(context)
        else:
            return self.value
