#-*- coding: UTF-8 -*-

from numba import jit
import numpy as np

@jit
def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

@jit
def l2_distance(X, Y, dim=1):
    return np.sqrt(np.sum(np.square(X - Y), axis=dim))