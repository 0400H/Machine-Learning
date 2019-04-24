#-*- coding: UTF-8 -*-

from numba import jit
import numpy as np

@jit
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))