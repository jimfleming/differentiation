from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensor import Tensor
from ops import AddOp, SubOp, MulOp, DivOp, DotOp, TransposeOp, SigmoidOp, SumOp, MeanOp, SquareOp, NegOp, AssignOp, \
    AbsOp, SignOp, PowerOp, LogOp, InvOp, GroupOp, ArgmaxOp, EqualOp, ReluOp, WhereOp, SoftmaxOp, GreaterOp, TileOp, ReshapeOp

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

    def log(self, x, name=None):
        op = LogOp(x, graph=self, name=name)
        return op.output

    def inv(self, x, name=None):
        op = InvOp(x, graph=self, name=name)
        return op.output

    def square(self, a, name=None):
        op = SquareOp(a, graph=self, name=name)
        return op.output

    def power(self, a, b, name=None):
        op = PowerOp(a, b, graph=self, name=name)
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

    def equal(self, a, b, name=None):
        op = EqualOp(a, b, graph=self, name=name)
        return op.output

    def argmax(self, a, axis=None, name=None):
        op = ArgmaxOp(a, axis=axis, graph=self, name=name)
        return op.output

    def transpose(self, a, axes=None, name=None):
        op = TransposeOp(a, axes=axes, graph=self, name=name)
        return op.output

    def sum(self, a, axis=None, name=None):
        op = SumOp(a, axis=axis, graph=self, name=name)
        return op.output

    def mean(self, a, axis=None, name=None):
        op = MeanOp(a, axis=axis, graph=self, name=name)
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

    def abs(self, x, name=None):
        op = AbsOp(x, graph=self, name=name)
        return op.output

    def sign(self, x, name=None):
        op = SignOp(x, graph=self, name=name)
        return op.output

    def softmax(self, x, name=None):
        op = SoftmaxOp(x, graph=self, name=name)
        return op.output

    def relu(self, x, name=None):
        op = ReluOp(x, graph=self, name=name)
        return op.output

    def where(self, condition, x, y, name=None):
        op = WhereOp(condition, x, y, graph=self, name=name)
        return op.output

    def greater(self, x, y, name=None):
        op = GreaterOp(x, y, graph=self, name=name)
        return op.output

    def group(self, inputs, name=None):
        op = GroupOp(inputs, graph=self, name=name)
        return op.output

    def tile(self, x, reps, name=None):
        op = TileOp(x, reps, graph=self, name=name)
        return op.output

    def reshape(self, x, shape, name=None):
        op = ReshapeOp(x, shape, graph=self, name=name)
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

        queue = []
        queue.append((y, 1))

        grads = {}
        while len(queue) > 0:
            y, grad_y = queue.pop(0)
            grad_y = self.ones_like(y) * self.convert(grad_y)

            gradients = y.op.gradient(grad_y)
            assert len(y.op.inputs) == len(gradients)

            for input_, grad in zip(y.op.inputs, gradients):
                assert np.array_equal(input_.shape, grad.shape), 'gradient shape much match input: {} != {}'.format(input_, grad)

                if input_ in grads:
                    grads[input_] += grad
                else:
                    grads[input_] = grad

                if not input_.op:
                    continue

                queue.append((input_, grad))

        return [grads[x] for x in xs]
