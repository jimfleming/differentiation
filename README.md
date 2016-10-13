# Implementing (parts of) TensorFlow (almost) from Scratch
## A walkthrough of symbolic differentiation

### Jim Fleming ([@jimmfleming](https://twitter.com/jimmfleming))

[main.py](main.html) |
[graph.py](graph.html) |
[tensor.py](tensor.html) |
[ops.py](ops.html) |
[session.py](session.html)

[Next: The Graph](graph.html)

This [literate programming](https://en.wikipedia.org/wiki/Literate_programming) exercise will construct a simple 2-layer feed-forward neural network to compute the [exclusive or](https://en.wikipedia.org/wiki/Exclusive_or), using [symbolic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) to compute the gradients automatically. In total, about 500 lines of code, including comments. The only functional dependency is numpy. I highly recommend reading Chris Olah's [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) for more background on what this code is doing.

The XOR task is convenient for a number of reasons: it's very fast to compute; it is not linearly separable thus requiring at least two layers and making the gradient calculation more interesting; it doesn't require more complicated matrix-matrix features such as broadcasting.

> (I'm also working on a more involved example for MNIST but as soon as I added support for matrix shapes and broadcasting the code ballooned by 5x and it was no longer a simple example.)

Let's start by going over the architecture. We're going to use four main components:

  - [`Graph`](graph.html), composed of `Tensor` nodes and `Op` nodes that together represent the computation we want to differentiate.
  - [`Tensor`](tensor.html) represents a value in the graph. Tensors keep a reference to the operation that produced it, if any.
  - [`BaseOp`](ops.html) represents a computation to perform and its differentiable components. Operations hold references to their input tensors and an output tensor.
  - [`Session`](session.html) is used to evaluate tensors in the graph.

**Note** the return from a graph operation is actually a tensor, representing the output of the operation.
