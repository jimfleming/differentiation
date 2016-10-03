from __future__ import print_function
from __future__ import division

import numpy as np

def one_hot(indices, num_labels):
    return np.eye(num_labels)[indices]

def shuffle(*args):
    p = np.random.permutation(len(args[0]))
    return [arg[p] for arg in args]

def softmax(x):
    exp_x = np.exp(x)
    assert np.all(np.isfinite(exp_x)), 'softmax exp'
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_d(x):
    return x * (1 - x)

def relu(x):
    return np.where(x > 0, x, 0)

def relu_d(x):
    return np.where(x > 0, 1, 0)

def he_init(fan_in, gain=1.0):
    return np.sqrt(gain / fan_in)

def l1(W):
    return np.sum(np.abs(W))

def l2(W):
    return np.sum(np.square(W))

def cross_entropy(y_preds, y_labels):
    return np.mean(-np.sum(y_labels * np.log(y_preds + 1e-7), axis=1))

def one_hot(indices, num_labels):
    return np.eye(num_labels, dtype=np.float32)[indices]

def label_accuracy(y_probs, y_labels):
    y_ = np.argmax(y_probs, axis=1)
    y = np.argmax(y_labels, axis=1)
    accuracy = np.mean(np.equal(y_, y))
    return accuracy

def regularize_parameters(params, regularizer):
    return np.sum([regularizer(param) for param in params])
