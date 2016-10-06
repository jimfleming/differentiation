from __future__ import print_function
from __future__ import division

import numpy as np

class BaseOp(object):
    """
    Represents a graph node that performs computation on tensors.
    For simplicity, every op has an array of N inputs and a M=1 outputs.
    """

    def __init__(self, inputs, graph=None, name=None):
        self.inputs = [graph.convert(input_) for input_ in inputs]
        self.output = graph.tensor(op=self, name=name+':0' if name is not None else 'None:0')
        self.graph = graph
        self.name = name

    def compute(self, context):
        raise NotImplementedError()

    def gradient(self, grad):
        # grad: N Tensor objects (grad w.r.t. each output)
        # return: M Tensor objects (partial grad w.r.t. each input)
        raise NotImplementedError()

    def __str__(self):
        return '{}("{}")'.format(type(self).__name__, self.name)

    def __repr__(self):
        return str(self)

class AddOp(BaseOp):

    def __init__(self, inputs, graph=None, name='Add'):
        super(AddOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] + context[b]

    def gradient(self, grad):
        grad = self.graph.convert(grad, name='grad_'+self.name)
        return [grad, grad]

class SubOp(BaseOp):

    def __init__(self, inputs, graph=None, name='Sub'):
        super(SubOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] - context[b]

    def gradient(self, grad):
        grad = self.graph.convert(grad, name='grad_'+self.name)
        return [grad, -grad]

class MulOp(BaseOp):

    def __init__(self, inputs, graph=None, name='Mul'):
        super(MulOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] * context[b]

    def gradient(self, grad):
        grad = self.graph.convert(grad, name='grad_'+self.name)
        a, b = self.inputs
        return [b * grad, a * grad]

class DivOp(BaseOp):

    def __init__(self, inputs, graph=None, name='Div'):
        super(DivOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] / context[b]

    def gradient(self, grad):
        grad = self.graph.convert(grad, name='grad_'+self.name)
        a, b = self.inputs
        return [grad / b, grad * (-a / self.graph.square(y))]

class SquareOp(BaseOp):

    def __init__(self, inputs, graph=None, name='Square'):
        super(SquareOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        input_, = self.inputs
        return context[input_]**2

    def gradient(self, grad):
        grad = self.graph.convert(grad, name='grad_'+self.name)
        return [2 * self.inputs[0] * grad]

class GradientOp(BaseOp):

    def __init__(self, y, x, grad_y=None, graph=None, name=None):
        self.grad_y = graph.convert(grad_y if grad_y else 1, name='grad_'+y.name)
        self.y = y
        self.x = x
        super(GradientOp, self).__init__([self.y, self.x, self.grad_y], graph, 'grad_'+x.name)

    def compute(self, context):
        # TODO: should be dumb: context should contain everything it needs to compute dy/dx
        return None

    def gradient(self, grad):
        raise NotImplementedError()
