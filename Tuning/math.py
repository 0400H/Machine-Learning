#-*- coding: UTF-8 -*-

from __future__ import division
from numba import jit
import numpy as np

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

def l2_distance(X, Y, dim=1):
    return np.sqrt(np.sum(np.square(X - Y), axis=dim))

def abs_error(func, *args):
    return np.abs(func(*args))