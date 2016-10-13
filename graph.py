"""
[main.py](main.html) |
[graph.py](graph.html) |
[tensor.py](tensor.html) |
[ops.py](ops.html) |
[session.py](session.html)

[Previous: Main](main.html) | [Next: Tensors](tensor.html)
"""

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

    def __init__(self):
        self.tensors = []

    def tensor(self, initial_value=None, op=None):
        """
        ## Graph#tensor
        Define a new tensor with the given initial_value and operation.
        """
        tensor = Tensor(initial_value=initial_value, graph=self, op=op)
        self.tensors.append(tensor)
        return tensor

    def convert(self, value):
        """
        ## Graph#convert
        Return the value if it is a `Tensor`, otherwise convert it to one.
        """
        if isinstance(value, Tensor):
            return value
        return self.tensor(initial_value=value)

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

    def group(self, inputs):
        op = GroupOp(inputs, graph=self)
        return op.output

    def gradients(self, y, xs):
        """
        ## Graph#gradients

        The `gradients` method performs backpropagation using [reverse accumulation](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation) and the [chain rule](https://en.wikipedia.org/wiki/Chain_rule#Higher_dimensions). It operates by traversing the graph from `y` to each `x` in `xs`, accumulating gradients, and returning the partial gradients for each `xs`. We use a queue to keep track of the next tensor for which to compute the gradient and keep a dictionary of the gradients computed thus far. Iteration starts from the target output `y` with an output gradient of 1.
        """

        queue = []
        queue.append((y, 1))

        grads = {}
        while len(queue) > 0:
            y, grad_y = queue.pop(0)
            grad_y = self.convert(grad_y)

            gradients = y.op.gradient(grad_y)
            assert len(gradients) == len(y.op.inputs)

            for tensor, gradient in zip(y.op.inputs, gradients):
                if tensor in grads:
                    grads[tensor] += gradient
                else:
                    grads[tensor] = gradient

                if tensor.op:
                    queue.append((tensor, gradient))

        return [grads[x] for x in xs]
