from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np; np.random.seed(67)

from tqdm import trange

from graph import Graph
from model import Model
from mnist import Dataset
from session import Session

"""
This literate programming exercise will construct a system for performing automatic differentiation as used in deep learning. In total about 500 lines of code, including comments.

We'll use the example task of learning a simple 2-layer feed-forward neural network to compute the [exclusive or](https://en.wikipedia.org/wiki/Exclusive_or) as a baseline to make sure everything is working as intended.

This task is convenient since it's very fast to compute, it is not linearly separable thus requiring at least two layers, and doesn't require much support for matrices (such as broadcasting).

> (I'm also working on a more involved example for MNIST but as soon as I added proper support for matrices the code ballooned by 5x and was no longer a simple example of symbolic differentiation. Adding support for shapes, broadcasting and such is actually much more work than the differentiation.)

Let's start by going over the architecture. We're going to use four main components:

  - A [`Graph`](graph.html), composed of `Tensor` nodes and `Op` nodes that combined represent the computation we want to differentiate.
  - A [`Tensor`](tensor.html) to represent a value in the graph. Tensors maintain a reference to the operation that produced it, if any.
  - An [`Op`](op.html) to represent a computation to perform and its differentiable components. Operations maintain references to their input tensors and an output tensor.
  - A [`Session`](session.html) to evaluate tensors in the graph.

**Note** the return from a graph operation is actually a tensor, representing the output of the operation.
"""

def main():
    graph = Graph()

    # Data
    X = graph.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    y = graph.tensor(np.array([[0, 1, 1, 0]])) # XOR

    # Model Parameters
    W0 = graph.tensor(np.random.normal(size=(2, 4)))
    b0 = graph.tensor(np.zeros(shape=(4,)))

    W1 = graph.tensor(np.random.normal(size=(4, 1)))
    b1 = graph.tensor(np.zeros(shape=(1,)))

    # Layer Activations
    h0 = graph.sigmoid(graph.dot(X, W0) + b0)
    h1 = graph.sigmoid(graph.dot(h0, W1) + b1)

    # [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) Loss Function
    loss_op = graph.mean(graph.square(graph.transpose(y) - h1))

    # Model Update
    parameters = [W0, b0, W1, b1]
    gradients = graph.gradients(loss_op, parameters)
    update_op = graph.group([
        graph.assign_sub(param, grad) \
            for param, grad in zip(parameters, gradients)
    ])

    # Training
    sess = Session(graph)
    with trange(10000) as pbar_epoch:
        for epoch in pbar_epoch:
            _, loss = sess.run([update_op, loss_op])
            pbar_epoch.set_description('loss: {:.8f}'.format(loss))

if __name__ == '__main__':
    main()
