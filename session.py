from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class Session(object):
    """Session performs the actual computation on a Graph."""

    def __init__(self, graph):
        self.graph = graph

    def run_op(self, op, context):
        # print('run_op', op)
        for input_ in op.inputs:
            if input_ not in context:
                context[input_] = self.eval_tensor(input_, context)
        return op.compute(context)

    def eval_tensor(self, tensor, context):
        # print('eval_tensor', tensor.op, tensor)
        if tensor not in context:
            if tensor.op is not None:
                context[tensor] = self.run_op(tensor.op, context)
            elif tensor.value is not None:
                context[tensor] = tensor.value
        if tensor not in context:
            raise ValueError('Tensor has no value: {}'.format(tensor))
        result = context[tensor]

        assert np.array_equal(tensor.shape, result.shape if isinstance(result, np.ndarray) else ()), 'Tensor value did not match Tensor shape: {} != {}'.format(tensor, result)

        return result

    def run(self, fetches, feed_dict=None):
        context = {}

        if feed_dict:
            context.update(feed_dict)

        for fetch in fetches:
            context[fetch] = self.eval_tensor(fetch, context)

        return [context[fetch] for fetch in fetches]
