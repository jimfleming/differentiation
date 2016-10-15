"""
[main.py](main.html) |
[graph.py](graph.html) |
[tensor.py](tensor.html) |
[ops.py](ops.html) |
[session.py](session.html)

[Previous: Operations](ops.html)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Session(object):
    """
    `Session` performs the actual computation on a Graph.
    """

    def __init__(self, graph):
        """
        Initializing a session with a graph and a state dictionary to hold
        tensor values.
        """
        self.graph = graph
        self.state = {}

    def run_op(self, op, context):
        """
        `run_op` takes as input an operation to run and a context to fetch
        pre-evaluted tensors.
        """
        args = [self.eval_tensor(tensor, context) for tensor in op.inputs]
        return op.compute(self, *args)

    def eval_tensor(self, tensor, context):
        """
        `eval_tensor` takes as input a tensor to evaluate and a context to
        fetch pre-evaluted tensors. If the tensor is not already in the context
        there are three possibilities for evaluating the tensor:

          - The tensor has an operation and is therefore the result of the
            operation that must be computed.
          - The tensor has an active state from another session run that can be
            fetched.
          - The tensor has an initial value from its instantiation that can be
            fetched and added to the state.
        """
        if tensor not in context:
            if tensor.op is not None:
                context[tensor] = self.run_op(tensor.op, context)
            elif tensor in self.state and self.state[tensor] is not None:
                context[tensor] = self.state[tensor]
            elif tensor not in self.state and tensor.initial_value is not None:
                context[tensor] = self.state[tensor] = tensor.initial_value

        return context[tensor]

    def run(self, tensors, feed_dict=None):
        """
        `run` takes a list of tensors to evaluate and a feed dictionary that
        can be used to override tensors.
        """
        context = {}

        if feed_dict:
            context.update(feed_dict)

        return [self.eval_tensor(tensor, context) for tensor in tensors]
