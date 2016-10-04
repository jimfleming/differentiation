from __future__ import print_function
from __future__ import division

class BaseOp(object):
    """Represents a graph node that performs computation on tensors."""

    def __init__(self, inputs, graph=None, name=None):
        self.inputs = [graph.convert(input_) for input_ in inputs]
        self.graph = graph
        self.name = name
        self.output = graph.tensor(op=self)

    def compute(self, context):
        raise NotImplementedError()

class AddOp(BaseOp):

    def __init__(self, inputs, graph=None, name=None):
        super(AddOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] + context[b]

class SubOp(BaseOp):

    def __init__(self, inputs, graph=None, name=None):
        super(SubOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] - context[b]

class MulOp(BaseOp):

    def __init__(self, inputs, graph=None, name=None):
        super(MulOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] * context[b]

class DivOp(BaseOp):

    def __init__(self, inputs, graph=None, name=None):
        super(DivOp, self).__init__(inputs, graph, name)

    def compute(self, context):
        a, b = self.inputs
        return context[a] / context[b]
