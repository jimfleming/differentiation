from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class BaseOp(object):
    """
    BaseOp represents an operation that performs computation on tensors.
    For simplicity, every op has an array of N `inputs` and M=1 `output`.
    """

    def __init__(self, inputs, graph):
        """
        Ensure all inputs are Tensors.
        Define a tensor to represent the result of the operation, which might be `None`.
        Keep a reference to the graph so that the operation can generate new operations in the gradient computation.
        """
        self.inputs = [graph.convert(input_) for input_ in inputs]
        self.output = graph.tensor(op=self)
        self.graph = graph

    def compute(self, *args):
        """
        The compute method receives as input the _evaluated_ input tensors and returns the
        result of performing its operation on the inputs.
        """
        raise NotImplementedError()

    def gradient(self, grad):
        """
        The gradient method computes the gradient of the output w.r.t. each of its inputs as new tensors.
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

class NegOp(BaseOp):

    def compute(self, x):
        return -x

    def gradient(self, grad):
        return [-grad]

class DotOp(BaseOp):

    def compute(self, a, b):
        return np.dot(a, b)

    def gradient(self, grad):
        a, b = self.inputs
        aT = self.graph.transpose(a)
        bT = self.graph.transpose(b)
        return [
            self.graph.dot(grad, bT),
            self.graph.dot(aT, grad),
        ]

class SquareOp(BaseOp):

    def compute(self, x):
        return np.square(x)

    def gradient(self, grad):
        x = self.inputs[0]
        return [grad * (2 * x)]

class TransposeOp(BaseOp):

    def __init__(self, inputs, graph):
        super(TransposeOp, self).__init__(inputs, graph)

    def compute(self, x):
        return np.transpose(x)

    def gradient(self, grad):
        return [self.graph.transpose(grad)]

class SigmoidOp(BaseOp):

    def compute(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, grad):
        y = self.output
        return [grad * (y * (1 - y))]

class MeanOp(BaseOp):

    def compute(self, x):
        return np.mean(x)

    def gradient(self, grad):
        # TODO:
        # input_tensor = self.inputs[0]
        # input_value = input_tensor.value
        # input_shape = input_value.shape
        factor = 4 # np.prod(input_shape)
        return [grad / factor]

class GroupOp(BaseOp):
    """
    The group operation exploits the fact that each input is automatically evaluated
    before computing the operation result.
    """

    def compute(self, *args):
        return None

    def gradient(self, grad):
        return [grad for inp in self.inpus]

class AssignOp(BaseOp):
    """
    The assign operation utilizes a receiving tensor and a new value tensor.
    """

    def compute(self, a, b):
        self.inputs[0].value = b
        return b
