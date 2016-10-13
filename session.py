from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Session(object):
    """`Session` performs the actual computation on a Graph."""

    def __init__(self, graph):
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
            elif tensor.value is not None:
                self.state[tensor] = tensor.value
                context[tensor] = self.state[tensor]

        if tensor not in context:
            raise ValueError('Tensor has no value: {}'.format(tensor))

        return context[tensor]

    def run(self, tensors, feed_dict=None):
        """
        `run` takes as input an array of tensors to evaluate and an initial context to fetch pre-evaluted tensors.
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
