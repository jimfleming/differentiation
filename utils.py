from __future__ import print_function
from __future__ import division

import numpy as np

def shuffle(*args):
    p = np.random.permutation(len(args[0]))
    return [arg[p] for arg in args]

def one_hot(indices, num_labels):
    return np.eye(num_labels, dtype=np.float32)[indices]

def reduced_shape(input_shape, axes, keep_dims):
    if axes is None:
        input_rank = len(input_shape)
        axes = list(range(input_rank))

    if not isinstance(axes, list):
        axes = [axes]

    if keep_dims:
        output_shape = list(input_shape)
        for axis in axes:
            output_shape[axis] = 1
    else:
        output_shape = list(input_shape)
        for axis in sorted(axes, reverse=True):
            del output_shape[axis]
    return output_shape

def safe_div(x, y):
    """Divide `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`."""
    return x // np.maximum(y, 1)
