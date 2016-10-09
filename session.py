from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

class Session(object):
    """Session performs the actual computation on a Graph."""

    def __init__(self, graph):
        self.graph = graph

    def run_op(self, op, feed_dict):
        context = {}
        for input_ in op.inputs:
            context[input_] = self.eval_tensor(input_, feed_dict)
        return op.compute(context)

    def eval_tensor(self, tensor, feed_dict):
        if feed_dict and tensor in feed_dict:
            tensor.value = feed_dict[tensor]
        elif tensor.op is not None:
            tensor.value = self.run_op(tensor.op, feed_dict)

        # if tensor.value is None:
        #     raise ValueError('You must feed a value for {} if it does not belong to an Op.'.format(tensor))

        return tensor.value

    def run(self, fetches, feed_dict=None):
        return [self.eval_tensor(fetch, feed_dict) for fetch in fetches]
