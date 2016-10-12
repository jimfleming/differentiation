from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensor import Tensor
from ops import AddOp, SubOp, MulOp, DivOp, DotOp, TransposeOp, SigmoidOp, MeanOp, SquareOp, NegOp, AssignOp, GroupOp

class Graph(object):
    """Graph represents a computation to be evaluated by a Session."""

    def tensor(self, value=None, op=None):
        return Tensor(value=value, graph=self, op=op)

    def convert(self, value):
        if isinstance(value, Tensor):
            return value
        return self.tensor(value=value)

    def add(self, a, b):
        op = AddOp([a, b], graph=self)
        return op.output

    def sub(self, a, b):
        op = SubOp([a, b], graph=self)
        return op.output

    def mul(self, a, b):
        op = MulOp([a, b], graph=self)
        return op.output

    def div(self, a, b):
        op = DivOp([a, b], graph=self)
        return op.output

    def square(self, x):
        op = SquareOp([x], graph=self)
        return op.output

    def sigmoid(self, x):
        op = SigmoidOp([x], graph=self)
        return op.output

    def dot(self, a, b):
        op = DotOp([a, b], graph=self)
        return op.output

    def transpose(self, x, axes=None):
        op = TransposeOp([x], axes=axes, graph=self)
        return op.output

    def mean(self, x, axis=None):
        op = MeanOp([x], axis=axis, graph=self)
        return op.output

    def assign(self, a, b):
        op = AssignOp([a, b], graph=self)
        return op.output

    def assign_add(self, a, b):
        op = AssignOp([a, a+b], graph=self)
        return op.output

    def assign_sub(self, a, b):
        op = AssignOp([a, a-b], graph=self)
        return op.output

    def abs(self, x):
        op = AbsOp([x], graph=self)
        return op.output

    def group(self, inputs):
        op = GroupOp(inputs, graph=self)
        return op.output

    def gradients(self, y, xs):
        """ Traverses graph from y to xs, accumulating gradients. """

        queue = []
        queue.append((y, 1))

        grads = {}
        while len(queue) > 0:
            y, grad_y = queue.pop(0)
            grad_y = self.convert(grad_y)

            gradients = y.op.gradient(grad_y)
            assert len(y.op.inputs) == len(gradients)

            for input_, grad in zip(y.op.inputs, gradients):
                if input_ in grads:
                    grads[input_] += grad
                else:
                    grads[input_] = grad

                if input_.op:
                    queue.append((input_, grad))

        return [grads[x] for x in xs]
