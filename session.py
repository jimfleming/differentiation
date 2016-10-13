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
        Initializing a session with a graph adds all of the graph's tensors to the internal state of the session.
        """
        self.graph = graph
        self.state = {}

    def run_op(self, op, context):
        """
        `run_op` takes as input an operation to run and a context to fetch pre-evaluted tensors.
        """
        args = []
        for tensor in op.inputs:
            if tensor not in context:
                context[tensor] = self.eval_tensor(tensor, context)
            args.append(context[tensor])

        return op.compute(self, *args)

    def eval_tensor(self, tensor, context):
        """
        `eval_tensor` takes as input a tensor to evaluate and a context to fetch pre-evaluted tensors.
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
        `run` takes as input a list of tensors to evaluate and an initial context to fetch pre-evaluted tensors.
        """
        context = {}

        if feed_dict:
            context.update(feed_dict)

        results = []
        for tensor in tensors:
            result = self.eval_tensor(tensor, context)
            results.append(result)

            context[tensor] = result

        return results
