# Implementing (parts of) TensorFlow (almost) from Scratch
## A Walkthrough of Symbolic Differentiation

This [literate programming](https://en.wikipedia.org/wiki/Literate_programming) exercise will construct a simple 2-layer feed-forward neural network to compute the [exclusive or](https://en.wikipedia.org/wiki/Exclusive_or), using [symbolic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) to compute the gradients automatically. In total, about 500 lines of code, including comments. The only functional dependency is numpy. I highly recommend reading Chris Olah's [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) for more background on what this code is doing.

The XOR task is convenient for a number of reasons: it's very fast to compute; it is not linearly separable thus requiring at least two layers and making the gradient calculation more interesting; it doesn't require more complicated matrix-matrix features such as broadcasting.

Let's start by going over the architecture. We're going to use four main components:

  - [`Graph`](graph.py), composed of `Tensor` nodes and `Op` nodes that together represent the computation we want to differentiate.
  - [`Tensor`](tensor.py) represents a value in the graph. Tensors keep a reference to the operation that produced it, if any.
  - [`BaseOp`](ops.py) represents a computation to perform and its differentiable components. Operations hold references to their input tensors and an output tensor.
  - [`Session`](session.py) is used to evaluate tensors in the graph.

**Note** the return from a graph operation is actually a tensor, representing the output of the operation.
