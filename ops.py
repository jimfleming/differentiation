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
        self.graph = graph

    def compute(self, context):
        raise NotImplementedError()

    def gradient(self, grad):
        raise NotImplementedError()

class AddOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(AddOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=a.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] + context[b]

    def gradient(self, grad):
        return [grad, grad]

class SubOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(SubOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] - context[b]

    def gradient(self, grad):
        return [grad, -grad]

class MulOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(MulOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] * context[b]

    def gradient(self, grad):
        a, b = self.inputs
        return [grad * b, grad * a]

class DivOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(DivOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] / context[b]

    def gradient(self, grad):
        a, b = self.inputs
        return [grad / b, grad * (-a / self.graph.square(b))]

class DotOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(DotOp, self).__init__([a, b], graph, name)
        assert self.inputs[0].shape[-1] == self.inputs[1].shape[0], 'inner dimensions must match in dot product: {} != {}'.format(a, b)
        self.output = graph.tensor(
            shape=[self.inputs[0].shape[0], self.inputs[1].shape[1]],
            op=self,
            name=self.name+'/output')

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

    def __init__(self, a, b, graph, name=None):
        super(PowerOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=a.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.power(a, b)

    def gradient(self, grad):
        a, b = self.inputs
        y = self.output
        return [grad * (b * a**(b-1)), grad * (y * self.graph.log(a))]

class LogOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(LogOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=x.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a = context[self.inputs[0]]
        return np.log(a)

    def gradient(self, grad):
        x = self.inputs[0]
        return [grad * self.graph.inv(x)]

class InvOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(InvOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=x.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return 1 / x

    def gradient(self, grad):
        x = self.inputs[0]
        y = self.output
        return [-grad * self.graph.inv(self.graph.square(x))]

class SquareOp(BaseOp):

    def __init__(self, a, graph, name=None):
        super(SquareOp, self).__init__([a], graph, name)
        self.output = graph.tensor(shape=a.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.square(x)

    def gradient(self, grad):
        return [grad * (2 * self.inputs[0])]

class TransposeOp(BaseOp):

    def __init__(self, a, graph, axes=None, name=None):
        super(TransposeOp, self).__init__([a], graph, name)
        self.output = graph.tensor(shape=np.roll(a.shape, 1), op=self, name=self.name+'/output')
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
        return [self.graph.transpose(grad, axes)]

class NegOp(BaseOp):

    def __init__(self, a, graph, name=None):
        super(NegOp, self).__init__([a], graph, name)
        self.output = graph.tensor(shape=a.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return -x

    def gradient(self, grad):
        return [-grad]

class AbsOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(AbsOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=x.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.abs(x)

    def gradient(self, grad):
        x = self.inputs[0]
        return [grad * self.graph.sign(x)]

class SignOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(SignOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=x.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.sign(x)

    def gradient(self, grad):
        x = self.inputs[0]
        return [self.graph.zeros(x.shape, name='sign_grad')]

class SigmoidOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(SigmoidOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=x.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return 1 / (1 + np.exp(-x))

    def gradient(self, grad):
        y = self.output
        return [grad * (y * (1 - y))]

class SumOp(BaseOp):

    def __init__(self, a, graph, axes=None, name=None):
        super(SumOp, self).__init__([a], graph, name)
        self.output = graph.tensor(shape=None, op=self, name=self.name+'/output')
        self.axes = axes

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.sum(x, self.axes)

    def gradient(self, grad):
        return [grad * self.graph.ones_like(self.inputs[0])]

class MeanOp(BaseOp):

    def __init__(self, a, graph, axes=None, name=None):
        super(MeanOp, self).__init__([a], graph, name)
        self.output = graph.tensor(shape=None, op=self, name=self.name+'/output')
        self.axes = axes

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.mean(x, self.axes)

    def gradient(self, grad):
        x = self.inputs[0]
        y = self.output

        factor = np.prod(x.shape) // np.maximum(np.prod(y.shape), 1)
        return [grad * self.graph.ones_like(x) / factor]

class GroupOp(BaseOp):

    def __init__(self, inputs, graph, axes=None, name=None):
        super(GroupOp, self).__init__(inputs, graph, name)
        self.output = graph.tensor(shape=None, op=self, name=self.name+'/output')

    def compute(self, context):
        return None

    def gradient(self, grad):
        return [grad for inp in self.inpus]

class AssignOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(AssignOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=None, op=self, name=self.name+'/output')

    def compute(self, context):
        self.inputs[0].value = context[self.inputs[1]]
        return None
