from __future__ import print_function
from __future__ import division

from collections import defaultdict

class Graph(object):

    def __init__(self):
        self.collections = defaultdict(list)
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)

    def add_edge(self, edge):
        self.edges.append(edge)

    def add_edges(self, edges):
        self.edges.extend(edges)

    def add_to_collection(self, key, value):
        self.collections[key].append(value)
