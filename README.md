# Implementing (parts of) TensorFlow (almost) from Scratch
## A Walkthrough of Symbolic Differentiation

This [literate programming](https://en.wikipedia.org/wiki/Literate_programming)
exercise will construct a simple 2-layer feed-forward neural network to compute
the [exclusive or](https://en.wikipedia.org/wiki/Exclusive_or), using [symbolic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) to
compute the gradients automatically. In total, about 500 lines of code,
including comments. The only functional dependency is numpy. I highly recommend
reading Chris Olah's [Calculus on Computational Graphs:
Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) for more
background on what this code is doing.
