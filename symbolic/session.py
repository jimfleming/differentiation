from __future__ import print_function
from __future__ import division

class Session(object):
    """A class for running operations."""

    def __init__(self, graph):
        self.graph = graph

    def run_one(self, fetch, feed_dict=None):
        if fetch.op:
            context = {}
            for input_ in fetch.op.inputs:
                if input_ in feed_dict:
                    input_eval = feed_dict[input_]
                else:
                    input_eval = self.run_one(input_, feed_dict)
                context[input_] = input_eval
            return fetch.op.compute(context)
        else:
            return fetch.value

    def run(self, fetches, feed_dict=None):
        return [self.run_one(fetch, feed_dict) for fetch in fetches]
