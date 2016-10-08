from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensor import Tensor
from ops import AddOp, SubOp, MulOp, DivOp, DotOp, TransposeOp, SigmoidOp, SumOp, MeanOp, SquareOp, NegOp, AssignOp

class Graph(object):
    """Graph represents a computation to be evaluated by a Session."""

    def tensor(self, value=None, shape=None, op=None, name=None):
        return Tensor(value=value, shape=shape, graph=self, op=op, name=name)

    def convert(self, value, name=None):
        if isinstance(value, Tensor):
            return value
        return self.tensor(value=value, name=name)

    def add(self, a, b, name=None):
        op = AddOp(a, b, graph=self, name=name)
        return op.output

    def sub(self, a, b, name=None):
        op = SubOp(a, b, graph=self, name=name)
        return op.output

    def mul(self, a, b, name=None):
        op = MulOp(a, b, graph=self, name=name)
        return op.output

    def div(self, a, b, name=None):
        op = DivOp(a, b, graph=self, name=name)
        return op.output

    def square(self, a, name=None):
        op = SquareOp(a, graph=self, name=name)
        return op.output

    def neg(self, a, name=None):
        op = NegOp(a, graph=self, name=name)
        return op.output

    def sigmoid(self, a, name=None):
        op = SigmoidOp(a, graph=self, name=name)
        return op.output

    def dot(self, a, b, name=None):
        op = DotOp(a, b, graph=self, name=name)
        return op.output

    def transpose(self, a, axes=None, name=None):
        op = TransposeOp(a, axes=axes, graph=self, name=name)
        return op.output

    def sum(self, a, axes=None, name=None):
        op = SumOp(a, axes=axes, graph=self, name=name)
        return op.output

    def mean(self, a, axes=None, name=None):
        op = MeanOp(a, axes=axes, graph=self, name=name)
        return op.output

    def assign(self, a, b, name=None):
        op = AssignOp(a, b, graph=self, name=name)
        return op.output

    def assign_add(self, a, b, name=None):
        op = AssignOp(a, a+b, graph=self, name=name)
        return op.output

    def assign_sub(self, a, b, name=None):
        op = AssignOp(a, a-b, graph=self, name=name)
        return op.output

    def ones(self, shape=None, name=None):
        return self.tensor(np.ones(shape=shape), name=name)

    def zeros(self, shape=None, name=None):
        return self.tensor(np.zeros(shape=shape), name=name)

    def ones_like(self, a, name=None):
        return self.tensor(np.ones(shape=a.shape), name=name)

    def zeros_like(self, a, name=None):
        return self.tensor(np.zeros(shape=a.shape), name=name)

    def random_normal(self, loc=0.0, scale=1.0, shape=None, name=None):
        return self.tensor(np.random.normal(loc=loc, scale=scale, size=shape), name=name)

    def random_uniform(self, low=0.0, high=1.0, shape=None, name=None):
        return self.tensor(np.random.uniform(low=low, high=high, size=shape), name=name)

    def gradients(self, y, xs, name=None):
        """ Traverses graph from y to xs, accumulating gradients. """

        grad_y = np.ones(shape=y.shape)
        grad_y = self.convert(grad_y)

        queue = []
        queue.append((y, grad_y))

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
