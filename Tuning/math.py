#-*- coding: UTF-8 -*-

from __future__ import division
from numba import jit
import numpy as np

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def softmax(X, dim=0):
    x_exp = np.exp(X)
    x_exp_sum = np.sum(x_exp, axis=dim)
    return x_exp / x_exp_sum

def l2_distance(X, Y, dim=1):
    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    if len(Y.shape) == 1:
        Y = Y.reshape(1,-1)
    return np.sqrt(np.sum(np.square(X - Y), axis=dim))

def abs_error(func, *args):
    return np.abs(func(*args))