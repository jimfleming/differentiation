from __future__ import print_function
from __future__ import division

from tensor import Tensor
from ops import AddOp, SubOp, MulOp, DivOp, GradientOp

class Graph(object):
    """A computation, represented as a dataflow graph."""

    def __init__(self):
        self.nodes = []

    def tensor(self, value=None, op=None, name=None):
        tensor = Tensor(value=value, graph=self, op=op, name=name)
        self.nodes.append(tensor)
        return tensor

    def convert(self, value):
        if isinstance(value, Tensor):
            return value
        return self.tensor(value=value)

    def add(self, a, b, name=None):
        op = AddOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def sub(self, a, b, name=None):
        op = SubOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def mul(self, a, b, name=None):
        op = MulOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def div(self, a, b, name=None):
        op = DivOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def square(self, a, name=None):
        op = SquareOp([a], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def gradient(self, y, x, name=None):
        op = GradientOp(y, x, graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def gradients(self, y, xs, name=None):
        return [self.gradient(y, x, name=name) for x in xs]
