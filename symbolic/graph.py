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

    def convert(self, value, name=None):
        if isinstance(value, Tensor):
            return value
        if name is None:
            name = 'tensor'
        return self.tensor(value=value, name=name)

    def add(self, a, b, name=None):
        if name is None:
            name = 'add'
        op = AddOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def sub(self, a, b, name=None):
        if name is None:
            name = 'sub'
        op = SubOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def mul(self, a, b, name=None):
        if name is None:
            name = 'mul'
        op = MulOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def div(self, a, b, name=None):
        if name is None:
            name = 'div'
        op = DivOp([a, b], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def square(self, a, name=None):
        if name is None:
            name = 'square'
        op = SquareOp([a], graph=self, name=name)
        self.nodes.append(op)
        return op.output

    def gradients(self, y, xs, name=None):
        queue = []
        queue.append((y, 1))

        grads = {}
        while len(queue) > 0:
            y, grad_y = queue.pop(0)

            for inp, grad in zip(y.op.inputs, y.op.gradient(grad_y)):
                if inp in grads:
                    grads[inp] += grad
                else:
                    grads[inp] = grad

                if not inp.op:
                    continue

                queue.append((inp, grad))

        return [grads[x] for x in xs]
