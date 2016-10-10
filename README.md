# Deep Learning from Scratch

No frameworks.

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
  - Implement Sum/Mean over specified axes
  - Investigate performance
  - Document with pycco
  - Simplify

## References

  - http://colah.github.io/posts/2015-08-Backprop/
  - http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network
  - https://en.wikipedia.org/wiki/Chain_rule
  - https://en.wikipedia.org/wiki/Backpropagation
  - https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation
  - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_grad.py
  - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_grad.py

## Explicitly Not Supported

  - Complex types
  - Data types other than float32
  - GPU/CUDA/CUDNN
  - Multi-threading
  - Performance
