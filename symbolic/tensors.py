from __future__ import print_function
from __future__ import division

class Variable(object):
    """Stateful data element in the graph."""

    def __init__(self, initial_value=None, name=None):
        pass

class Placeholder(object):
    """Data element in the graph which must be specified at run-time."""

    def __init__(self, shape=None, name=None):
        pass

class Constant(object):
    """Immutable data element in the graph."""

    def __init__(self, value):
        pass
