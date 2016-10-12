from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from utils import reduced_shape, safe_div, broadcast_shape

class BaseOp(object):
    """
    BaseOp represents an operation that performs computation on tensors.
    For simplicity, every op has an array of N inputs and a M=1 outputs.
    """

    def __init__(self, inputs, graph, name=None):
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name

        # ensure all inputs are Tensors
        self.inputs = [graph.convert(input_) for input_ in inputs]
        self.graph = graph

    def compute(self, context):
        raise NotImplementedError()

    def gradient(self, grad):
        """
        BaseOp#gradient computes the gradient of the output w.r.t. each of its inputs.
        Given a gradient w.r.t. to the operation's output, returns the partial gradients w.r.t. each input.
        """
        raise NotImplementedError()

    def __repr__(self):
        return '{}("{}")'.format(type(self).__name__, self.name)

class AddOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(AddOp, self).__init__([a, b], graph, name)
        output_shape = broadcast_shape(self.inputs[0].shape, self.inputs[1].shape)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] + context[b]

    def gradient(self, grad):
        # x = self.inputs[0]
        # y = self.inputs[1]
        # sx = x.shape
        # sy = y.shape
        # rx, ry = broadcast_gradient_args(sx, sy)
        # return [self.graph.reshape(self.graph.sum(grad, axis=rx), sx),
        #         self.graph.reshape(self.graph.sum(grad, axis=ry), sy)]
        return [grad * self.graph.ones_like(self.inputs[0]), grad * self.graph.ones_like(self.inputs[1])]

class SubOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(SubOp, self).__init__([a, b], graph, name)
        output_shape = broadcast_shape(self.inputs[0].shape, self.inputs[1].shape)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] - context[b]

    def gradient(self, grad):
        return [grad * self.graph.ones_like(self.inputs[0]), -grad * self.graph.ones_like(self.inputs[1])]

class MulOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(MulOp, self).__init__([a, b], graph, name)
        output_shape = broadcast_shape(self.inputs[0].shape, self.inputs[1].shape)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] * context[b]

    def gradient(self, grad):
        a, b = self.inputs
        return [grad * b, grad * a]

class DivOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(DivOp, self).__init__([a, b], graph, name)
        output_shape = broadcast_shape(self.inputs[0].shape, self.inputs[1].shape)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a, b = self.inputs
        return context[a] / context[b]

    def gradient(self, grad):
        a, b = self.inputs
        return [grad / b, grad * (-a / self.graph.square(b))]

class DotOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(DotOp, self).__init__([a, b], graph, name)
        assert self.inputs[0].shape[-1] == self.inputs[1].shape[0], 'inner dimensions must match in dot product: {} != {}'.format(a, b)
        self.output = graph.tensor(
            shape=[self.inputs[0].shape[0], self.inputs[1].shape[1]],
            op=self,
            name=self.name+'/output')

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.dot(a, b)

    def gradient(self, grad):
        aT = self.graph.transpose(self.inputs[0])
        bT = self.graph.transpose(self.inputs[1])
        return [
            self.graph.dot(grad, bT),
            self.graph.dot(aT, grad),
        ]

class ArgmaxOp(BaseOp):

    def __init__(self, x, graph, axis=None, name=None):
        super(ArgmaxOp, self).__init__([x], graph, name)
        input_shape = self.inputs[0].shape
        output_shape = reduced_shape(input_shape, axis, keep_dims=False)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')
        self.axis = axis

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.argmax(x, axis=self.axis)

class EqualOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(EqualOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.equal(a, b)

class PowerOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(PowerOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.power(a, b)

    def gradient(self, grad):
        a, b = self.inputs
        y = self.output
        return [grad * (b * a**(b-1)), grad * (y * self.graph.log(a))]

class LogOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(LogOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a = context[self.inputs[0]]
        return np.log(a)

    def gradient(self, grad):
        x = self.inputs[0]
        return [grad * self.graph.inv(x)]

class InvOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(InvOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return 1 / x

    def gradient(self, grad):
        x = self.inputs[0]
        y = self.output
        return [-grad * self.graph.inv(self.graph.square(x))]

class SquareOp(BaseOp):

    def __init__(self, a, graph, name=None):
        super(SquareOp, self).__init__([a], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.square(x)

    def gradient(self, grad):
        return [grad * (2 * self.inputs[0])]

class TransposeOp(BaseOp):

    def __init__(self, a, graph, axes=None, name=None):
        super(TransposeOp, self).__init__([a], graph, name)
        input_shape = self.inputs[0].shape
        output_shape = np.roll(input_shape, 1)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')
        if axes is None:
            self.axes = [axis for axis, size in enumerate(self.inputs[0].shape)]
        else:
            self.axes = axes

    def compute(self, context):
        x = context[self.inputs[0]]
        axes = np.roll(self.axes, 1)
        return np.transpose(x, axes)

    def gradient(self, grad):
        axes = np.argsort(self.axes)
        return [self.graph.transpose(grad, axes)]

class NegOp(BaseOp):

    def __init__(self, a, graph, name=None):
        super(NegOp, self).__init__([a], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return -x

    def gradient(self, grad):
        return [-grad]

class WhereOp(BaseOp):

    def __init__(self, condition, x, y, graph, name=None):
        super(WhereOp, self).__init__([condition, x, y], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        condition = context[self.inputs[0]]
        x = context[self.inputs[1]]
        y = context[self.inputs[2]]
        return np.where(condition, x, y)

class GreaterOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(GreaterOp, self).__init__([a, b], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        a = context[self.inputs[0]]
        b = context[self.inputs[1]]
        return np.greater(a, b)

class ReluOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(ReluOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.maximum(0, x)

    def gradient(self, grad):
        y = self.output
        return [grad * self.graph.where(y > 0, self.graph.ones_like(y), self.graph.zeros_like(y))]

class SoftmaxOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(SoftmaxOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        exp_x = np.exp(x)
        assert np.all(np.isfinite(exp_x)), 'softmax exp must be finite'
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def gradient(self, grad):
        y = self.output
        return [grad * (y * (1 - y))]

class AbsOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(AbsOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.abs(x)

    def gradient(self, grad):
        x = self.inputs[0]
        return [grad * self.graph.sign(x)]

class SignOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(SignOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.sign(x)

    def gradient(self, grad):
        x = self.inputs[0]
        return [self.graph.zeros(x.shape, name='sign_grad')]

class SigmoidOp(BaseOp):

    def __init__(self, x, graph, name=None):
        super(SigmoidOp, self).__init__([x], graph, name)
        self.output = graph.tensor(shape=self.inputs[0].shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return 1 / (1 + np.exp(-x))

    def gradient(self, grad):
        y = self.output
        return [grad * (y * (1 - y))]

class ReshapeOp(BaseOp):

    def __init__(self, x, shape, graph, name=None):
        super(ReshapeOp, self).__init__([x], graph, name)
        self.shape = shape
        self.output = graph.tensor(shape=self.shape, op=self, name=self.name+'/output')

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.reshape(x, self.shape)

class TileOp(BaseOp):

    def __init__(self, x, reps, graph, name=None):
        super(TileOp, self).__init__([x], graph, name)
        input_shape = self.inputs[0].shape
        output_shape = [dim * reps[i] for i, dim in enumerate(input_shape)]
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')
        self.reps = reps

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.tile(x, self.reps)

class SumOp(BaseOp):

    def __init__(self, x, graph, axis=None, name=None):
        super(SumOp, self).__init__([x], graph, name)
        input_shape = self.inputs[0].shape
        output_shape = reduced_shape(input_shape, axis, keep_dims=False)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')
        self.axis = axis

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.sum(x, axis=self.axis)

    @staticmethod
    def gradient_static(op, grad):
        input_shape = op.inputs[0].shape
        output_shape_kept_dims = reduced_shape(input_shape, op.axis, keep_dims=True)
        tile_scaling = safe_div(input_shape, output_shape_kept_dims)
        grad = op.graph.reshape(grad, output_shape_kept_dims)
        tile = op.graph.tile(grad, tile_scaling)
        return [tile]

    def gradient(self, grad):
        return SumOp.gradient_static(self, grad)

class MeanOp(BaseOp):

    def __init__(self, a, graph, axis=None, name=None):
        super(MeanOp, self).__init__([a], graph, name)
        input_shape = self.inputs[0].shape
        output_shape = reduced_shape(input_shape, axis, keep_dims=False)
        self.output = graph.tensor(shape=output_shape, op=self, name=self.name+'/output')
        self.axis = axis

    def compute(self, context):
        x = context[self.inputs[0]]
        return np.mean(x, self.axis)

    def gradient(self, grad):
        sum_grad = SumOp.gradient_static(self, grad)[0]

        input_shape = self.inputs[0].shape
        output_shape = self.output.shape
        factor = safe_div(np.prod(input_shape), np.prod(output_shape))

        return [sum_grad / factor]

class GroupOp(BaseOp):

    def __init__(self, inputs, graph, axes=None, name=None):
        super(GroupOp, self).__init__(inputs, graph, name)
        self.output = graph.tensor(shape=(), op=self, name=self.name+'/output')

    def compute(self, context):
        return None

    def gradient(self, grad):
        return [grad for inp in self.inpus]

class AssignOp(BaseOp):

    def __init__(self, a, b, graph, name=None):
        super(AssignOp, self).__init__([a, b], graph, name)
        assert np.array_equal(self.inputs[0].shape, self.inputs[1].shape), 'Tensors must have the same shape: {} != {}'.format(self.inputs[0], self.inputs[1])
        self.output = graph.tensor(shape=(), op=self, name=self.name+'/output')

    def compute(self, context):
        # TODO: use context, not value which should just be init
        self.inputs[0].value = context[self.inputs[1]]
        return None
