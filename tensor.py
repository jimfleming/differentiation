"""
[main.py](main.html) |
[graph.py](graph.html) |
[tensor.py](tensor.html) |
[ops.py](ops.html) |
[session.py](session.html)

[Previous: The Graph](graph.html) | [Next: Operations](ops.html)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Tensor(object):
    """
    `Tensor` represents a _value_ in the graph. It's just a data container with
    methods for operator overloading (each of which delegate to the graph). It
    includes:

      - The initial value of the tensor.
      - The operation which produced the tensor, if applicable.
      - A reference to the graph this tensor belongs to.
    """

    def __init__(self, initial_value, op, graph):
        self.initial_value = initial_value
        self.graph = graph
        self.op = op

    # ## [Operator Overloading](https://docs.python.org/2/reference/datamodel.html?highlight=__radd__#emulating-numeric-types)
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

    # ## [Reverse Operator Overloading](https://docs.python.org/2/reference/datamodel.html?highlight=__radd__#object.__radd__)
    def __radd__(self, other):
        return self.graph.add(other, self)

    def __rsub__(self, other):
        return self.graph.sub(other, self)

    def __rmul__(self, other):
        return self.graph.mul(other, self)

    def __rtruediv__(self, other):
        return self.graph.div(other, self)
