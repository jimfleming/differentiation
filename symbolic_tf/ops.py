from __future__ import print_function
from __future__ import division

class Add(object):

    def __init__(self, *args):
        self.inputs = args

    def __call__(self):
        return self.inputs[0] + self.inputs[1]

    def gradient(self, grad):
        return [(self.inputs[0], grad), (self.inputs[0], grad)]

class Sub(object):

    def __init__(self, a, b):
        pass

    def __call__(self):
        return tensor_result

class Mul(object):

    def __init__(self, a, b):
        pass

    def __call__(self):
        return tensor_result

class Div(object):

    def __init__(self, a, b):
        pass

    def __call__(self):
        return tensor_result

class Dot(object):

    def __init__(self, a, b):
        pass

    def __call__(self):
        return tensor_result
