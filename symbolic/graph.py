from __future__ import print_function
from __future__ import division

from tensor import Tensor
from ops import AddOp, SubOp, MulOp, DivOp

class Graph(object):
    """A computation, represented as a dataflow graph."""

    def __init__(self):
        self.nodes = []

    def tensor(self, value=None, op=None, name=None):
        tensor = Tensor(value=value, graph=self, op=op, name=name)
        self.nodes.append(tensor)
        return tensor

    def add(self, inputs, name=None):
        op = AddOp(inputs, graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def sub(self, inputs, name=None):
        op = SubOp(inputs, graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def mul(self, inputs, name=None):
        op = MulOp(inputs, graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def div(self, inputs, name=None):
        op = DivOp(inputs, graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def convert(self, value):
        if isinstance(value, Tensor):
            return value
        return self.tensor(value=value, op=None, name=None)
