from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

class Session(object):
    """Session performs the actual computation on a Graph."""

    def __init__(self, graph):
        self.graph = graph

    def run_op(self, op, feed_dict):
        input_eval = {}
        for input_ in op.inputs:
            input_eval[input_] = self.eval_tensor(input_, feed_dict)
        return op.compute(input_eval)

    def eval_tensor(self, tensor, feed_dict):
        if feed_dict and tensor in feed_dict:
            return feed_dict[tensor]
        elif tensor.value is not None:
            return tensor.value
        elif tensor.op is not None:
            return self.run_op(tensor.op, feed_dict)
        else:
            raise Exception('Invalid Tensor: absent from feed and no value or op')

    def run(self, fetches, feed_dict=None):
        return [self.eval_tensor(fetch, feed_dict) for fetch in fetches]
