from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Tensor(object):
    """
    `Tensor` represents a _value_ in the graph. Its just a data container with methods for operator overloading which delegate to the graph. It includes:

      - The represented value of the tensor (only if it is not the result of an operation; e.g. it was initialized with a value.)
      - A reference to the graph this tensor belongs to.
      - The operation which produced the tensor, if applicable.

    **Note** that unlike TensorFlow, the current value of a tensor is held in the graph, not in the session, unless that tensor is a output for a operation, then its value is held in the session's context.
    """

    def __init__(self, value, op, graph):
        self.value = value
        self.graph = graph
        self.op = op

    # Operator overloading:
    def __add__(self, other):
        return self.graph.add(self, other)

    def __sub__(self, other):
        return self.graph.sub(self, other)

    def __mul__(self, other):
        return self.graph.mul(self, other)

    def __truediv__(self, other):
        return self.graph.div(self, other)

    def __neg__(self):
        return self.graph.neg(self)

    # Reverse operator overloading:
    def __radd__(self, other):
        return self.graph.add(other, self)

    def __rsub__(self, other):
        return self.graph.sub(other, self)

    def __rmul__(self, other):
        return self.graph.mul(other, self)

    def __rtruediv__(self, other):
        return self.graph.div(other, self)
