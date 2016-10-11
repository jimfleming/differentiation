# Deep Learning from Scratch

No frameworks.

## Overview

  - Graph
  - Tensor
  - Ops
  - Session

## Process

### Step 1. Simple Linear Classifier (AND/OR)

  - Linear problem (AND/OR) with a single layer classifier
  - Ones initialization
  - Manual gradients

### Step 2. Automatic Diff

  - Automatic gradients
  - Individual gradient ops

### Step 2. XOR (multiple layers)

  - Random normal init

### Step 3. Regularization, softmax, cross entropy loss

  - More complicated loss and activation fns

### Step 4. MNIST

## TODO

  - Solve MNIST
  - Fix assigning value
  - Document with pycco
  - Fill out README
  - Simplify

## References

  - Backprop
    - http://colah.github.io/posts/2015-08-Backprop/
    - http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network
    - https://en.wikipedia.org/wiki/Backpropagation
    - https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation
  - Math
    - https://en.wikipedia.org/wiki/Chain_rule
    - https://en.wikipedia.org/wiki/Product_rule
    - https://en.wikipedia.org/wiki/Sum_rule_in_differentiation
    - https://en.wikipedia.org/wiki/Derivative#Rules_of_computation
    - https://en.wikipedia.org/wiki/Differentiation_rules
  - TensorFlow Gradients
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_ops.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_grad.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_grad.py

## Explicitly Not Supported

  - Complex numbers
  - Sophisticated handling of data types, slicing or broadcasting
  - GPU/CUDA/CUDNN
  - Multi-threading
