from __future__ import print_function
from __future__ import division

class Session(object):

    def __init__(self, graph):
        self.graph = graph
        self.state = {}

    def run(self, op, feed_dict):
        # TODO: evaluate input nodes to op, then op
        op(feed_dict)

    def close(self):
        pass
