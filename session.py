from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

class Session(object):
    """Session performs the actual computation on a Graph."""

    def __init__(self, graph):
        self.graph = graph
        self.context = {}

    def run_op(self, op, feed_dict):
        for input_ in op.inputs:
            self.eval_tensor(input_, feed_dict)
        return op.compute(self.context)

    def eval_tensor(self, tensor, feed_dict):
        if tensor in self.context:
            return self.context[tensor]

        if feed_dict and tensor in feed_dict:
            self.context[tensor] = feed_dict[tensor]
        elif tensor.value is not None:
            self.context[tensor] = tensor.value
        elif tensor.op is not None:
            self.context[tensor] = self.run_op(tensor.op, feed_dict)
        else:
            raise ValueError('You must feed a value for {} if it does not belong to an Op.'.format(tensor))

        return self.context[tensor]

    def run(self, fetches, feed_dict=None):
        return [self.eval_tensor(fetch, feed_dict) for fetch in fetches]
