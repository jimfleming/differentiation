from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class BaseOp(object):
    """
    BaseOp represents an operation that performs computation on tensors.
    For simplicity, every op has an array of N inputs and a M=1 outputs.
    """

    def __init__(self, inputs, graph):
        # ensure all inputs are Tensors
        self.inputs = [graph.convert(input_) for input_ in inputs]
        self.output = graph.tensor(op=self)
        self.graph = graph

    def compute(self, context):
        raise NotImplementedError()

    def gradient(self, grad):
        """
        BaseOp#gradient computes the gradient of the output w.r.t. each of its inputs.
        Given a gradient w.r.t. to the operation's output, returns the partial gradients w.r.t. each input.
        """
        raise NotImplementedError()

class AddOp(BaseOp):

    def compute(self, a, b):
        return a + b

    def gradient(self, grad):
        return [grad, grad]

class SubOp(BaseOp):

    def compute(self, a, b):
        return a - b

    def gradient(self, grad):
        return [grad, -grad]

class MulOp(BaseOp):

    def compute(self, a, b):
        return a * b

    def gradient(self, grad):
        a, b = self.inputs
        return [grad * b, grad * a]

class DivOp(BaseOp):

    def compute(self, a, b):
        return a / b

    def gradient(self, grad):
        a, b = self.inputs
        return [grad / b, grad * (-a / self.graph.square(b))]

class DotOp(BaseOp):

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.dot(a, b)

    def gradient(self, grad):
        aT = self.graph.transpose(self.inputs[0])
        bT = self.graph.transpose(self.inputs[1])
        return [
            self.graph.dot(grad, bT),
            self.graph.dot(aT, grad),
        ]

class PowerOp(BaseOp):

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.power(a, b)

    def gradient(self, grad):
        a, b = self.inputs
        y = self.output
        return [grad * (b * a**(b-1)), grad * (y * self.graph.log(a))]

class LogOp(BaseOp):

    def compute(self, context):
        a = context[self.inputs[0]]
        return np.log(a)

    def gradient(self, grad):
        x = self.inputs[0]
        return [grad * self.graph.inv(x)]

class InvOp(BaseOp):

    def compute(self, context):
        x = context[self.inputs[0]]
        return 1 / x

    def gradient(self, grad):
        x = self.inputs[0]
        y = self.output
        return [-grad * self.graph.inv(self.graph.square(x))]

class SquareOp(BaseOp):

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.square(x)

    def gradient(self, grad):
        return [grad * (2 * self.inputs[0])]

class TransposeOp(BaseOp):

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.transpose(x, axes)

    def gradient(self, grad):
        axes = np.argsort(self.axes)
        return [self.graph.transpose(grad, axes)]

class NegOp(BaseOp):

    def compute(self, context):
        x = context[self.inputs[0]]
        return -x

    def gradient(self, grad):
        return [-grad]

class AbsOp(BaseOp):

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.abs(x)

    def gradient(self, grad):
        x = self.inputs[0]
        return [grad * self.graph.sign(x)]

class SigmoidOp(BaseOp):

    def compute(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, grad):
        y = self.output
        return [grad * (y * (1 - y))]

class MeanOp(BaseOp):

    def compute(self, x):
        return np.mean(x, self.axis)

    def gradient(self, grad):
        input_shape = self.inputs[0].shape
        factor = np.prod(input_shape)
        return [grad / factor]

class GroupOp(BaseOp):

    def compute(self, context):
        return None

    def gradient(self, grad):
        return [grad for inp in self.inpus]

class AssignOp(BaseOp):

    def compute(self, context):
        # TODO: use context, not value which should just be init
        self.inputs[0].value = context[self.inputs[1]]
        return None
