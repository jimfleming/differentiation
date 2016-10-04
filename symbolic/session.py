from __future__ import print_function
from __future__ import division

class Session(object):
    """A class for running operations."""

    def __init__(self, graph):
        self.graph = graph

    def run(self, fetches, feed_dict=None):
        results = []
        for fetch in fetches:
            if fetch.op:
                context = {}
                for input_ in fetch.op.inputs:
                    if input_ in feed_dict:
                        input_eval = feed_dict[input_]
                    else:
                        input_eval = self.run([input_], feed_dict)[0]
                    context[input_] = input_eval
                result = fetch.op.compute(context)
            else:
                result = fetch.value
            results.append(result)
        return results
