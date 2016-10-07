from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class BaseOp(object):
    """
    BaseOp represents an operation that performs computation on tensors.
    For simplicity, every op has an array of N inputs and a M=1 outputs.
    """

    def __init__(self, inputs, graph, name=None):
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

        self.inputs = [graph.convert(inp) for inp in inputs]
        self.output = graph.tensor(op=self, name=self.name+'/output')
        self.graph = graph

    def compute(self, context):
        raise NotImplementedError()

    def gradient(self, grad):
        # grad: gradient w.r.t. output
        # returns: partial gradient w.r.t. each input
        raise NotImplementedError()

    def __str__(self):
        return '{}("{}")'.format(type(self).__name__, self.name)

    def __repr__(self):
        return str(self)

class AddOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(AddOp, self).__init__([a, b], graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] + context[b]

    def gradient(self, grad):
        return [grad, grad]

class SubOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(SubOp, self).__init__([a, b], graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] - context[b]

    def gradient(self, grad):
        return [grad, -grad]

class MulOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(MulOp, self).__init__([a, b], graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] * context[b]

    def gradient(self, grad):
        a, b = self.inputs
        return [b * grad, a * grad]

class DivOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(DivOp, self).__init__([a, b], graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] / context[b]

    def gradient(self, grad):
        a, b = self.inputs
        return [grad / b, grad * (-a / self.graph.square(y))]

class DotOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(DotOp, self).__init__([a, b], graph, name)

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.dot(a, b)

    def gradient(self, grad):
        return [
            self.graph.dot(grad, self.graph.transpose(self.inputs[1])),
            self.graph.dot(self.graph.transpose(self.inputs[0]), grad),
        ]

class SquareOp(BaseOp):

    def __init__(self, a, graph, name=None):
        super(SquareOp, self).__init__([a], graph, name)

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.square(x)

    def gradient(self, grad):
        return [2 * self.inputs[0] * grad]

class TransposeOp(BaseOp):

    def __init__(self, a, graph, axes=None, name=None):
        super(TransposeOp, self).__init__([a], graph, name)
        if axes is None:
            self.axes = [axis for axis, size in enumerate(a.shape)]
        else:
            self.axes = axes

    def compute(self, context):
        x = context[self.inputs[0]]
        axes = np.roll(self.axes, 1)
        return np.transpose(x, axes)

    def gradient(self, grad):
        axes = np.argsort(self.axes)
        return [self.graph.transpose(self.inputs[0], axes)]

class NegOp(BaseOp):

    def __init__(self, a, graph, name=None):
        super(NegOp, self).__init__([a], graph, name)

    def compute(self, context):
        x = context[self.inputs[0]]
        return -x

    def gradient(self, grad):
        return [-grad]

class SigmoidOp(BaseOp):

    def __init__(self, a, graph, name=None):
        super(SigmoidOp, self).__init__([a], graph, name)

    def compute(self, context):
        x = context[self.inputs[0]]
        return 1 / (1 + np.exp(-x))

    def gradient(self, grad):
        return [grad * (1 - grad)]

class SumOp(BaseOp):

    def __init__(self, a, graph, axes=None, name=None):
        super(SumOp, self).__init__([a], graph, name)
        self.axes = axes

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.sum(x, self.axes)

    def gradient(self, grad):
        # vector of partial derviatives
        # print(self.inputs[0].shape, self.axes)
        return [self.graph.tensor(grad, shape=self.inputs[0].shape)]

class MeanOp(BaseOp):

    def __init__(self, a, axes, graph, name=None):
        super(MeanOp, self).__init__([a], graph, name)
        self.axes = axes

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.mean(x, self.axes)

    def gradient(self, grad):
        return [self.graph.tensor(np.prod(self.inputs[0].shape)) * self.graph.tensor(grad, shape=self.inputs[0].shape)]

class AssignOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(AssignOp, self).__init__([a, b], graph, name)

    def compute(self, context):
        context[self.inputs[0]] = context[self.inputs[1]]
        return None
