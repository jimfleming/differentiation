from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensor import Tensor
from ops import AddOp, SubOp, MulOp, DivOp, \
                DotOp, TransposeOp, SquareOp, NegOp, \
                MeanOp, SigmoidOp, AssignOp, GroupOp

class Graph(object):
    """
    `Graph` represents a computation to be evaluated by a `Session`. With the exception of `Graph#tensor`, `Graph#convert`, and `Graph#gradients`, most methods simply create an operation and return the output tensor of the operation.
    """

    def tensor(self, value=None, op=None):
        """
        ## Graph#tensor
        Define a new tensor with the given value and operation.
        """
        return Tensor(value=value, graph=self, op=op)

    def convert(self, value):
        """
        ## Graph#convert
        Return the value if it is a `Tensor`, otherwise convert it to one.
        """
        if isinstance(value, Tensor):
            return value
        return self.tensor(value=value)

    def add(self, a, b):
        """
        ## Graph#[add|sub|mul|div|...]
        Define a new operation with the provided input tensors and return the operations' output.
        """
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

    def neg(self, x):
        op = NegOp([x], graph=self)
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

    def transpose(self, x):
        op = TransposeOp([x], graph=self)
        return op.output

    def mean(self, x):
        op = MeanOp([x], graph=self)
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

    def group(self, inputs):
        op = GroupOp(inputs, graph=self)
        return op.output

    def gradients(self, y, xs):
        """
        ## Graph#gradients

        Traverse the graph from y to xs, accumulating gradients then return the gradients for each x in xs.
        Use a queue to keep track of the next tensor for which to compute the gradient.
        (Alternatively we could use recursion but this felt cleaner.)
        Start from the target output `y` with an output gradient of 1.
        We keep a dictionary of the gradients computed this far. Gradients accumulate for tensors already in the dictionary.
        Using the given output gradient, compute the partial derivative of the op w.r.t. inputs.
        Iterate through each input and gradient pair, accumulating the gradients for the input.
        If the input has an op (e.g. it's not a variable but an output tensor) add the input and its gradient to the queue.
        """

        queue = []
        queue.append((y, 1))

        grads = {}
        while len(queue) > 0:
            y, grad_y = queue.pop(0)
            grad_y = self.convert(grad_y)

            gradients = y.op.gradient(grad_y)
            assert len(gradients) == len(y.op.inputs)

            for input_, grad in zip(y.op.inputs, gradients):
                if input_ in grads:
                    grads[input_] += grad
                else:
                    grads[input_] = grad

                if input_.op:
                    queue.append((input_, grad))

        return [grads[x] for x in xs]
