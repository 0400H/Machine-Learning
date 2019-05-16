#-*- coding: UTF-8 -*-

import os, sys
try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'KNN/'
    sys.path.append(__ML_PATH__)
    from Tuning.datatune import *
except ModuleNotFoundError:
    __F_PATH__ = os.getcwd() + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../')
    pass
__ALGO_PATH__ = __F_PATH__
sys.path.append(__ML_PATH__)

from Tuning.datatune import *
from Tuning.math import *
from numba import jit
import numpy as np
import sympy

class symbol_compute(object):
    def s_init(self, s_func, *s_args):
        self.s_args = s_args
        self.s_func = s_func

    def s_dfunc(self, s_arg):
        return sympy.diff(self.s_func, s_arg)

    def v_func(self, *v_args):
        args_map = {}
        args_length = len(self.s_args)
        for index in range(args_length):
            args_map[self.s_args[index]] = v_args[index]
            # print(self.s_args[index], v_args[index])
        return self.s_func.evalf(subs=args_map)

    def v_dfunc(self, s_dfunc, s_arg, v_arg):
        return s_dfunc.evalf(subs={s_arg: v_arg})
    pass

@jit
def gradient_descent(f, df, x0, learn_rate=0.2, precision=1e-6):
    '''
        f: loss function
        df: loss function derivative
        X_(n+1) = X_(n) - lr * df(X)
    '''
    iter_num = 0
    x_iter = x0
    while abs_error(f, x_iter) > precision:
        iter_num += 1
        x_iter = x_iter - learn_rate * df(x_iter)
        print(iter_num, x_iter, df(x_iter), f(x_iter), abs_error(f, x_iter) > precision)
    return x_iter, iter_num

@jit
def newtons_iteration(f, df, x0, precision=1e-6):
    '''
        f: loss function
        df: loss function derivative
        X_(n+1) = X_(n) - f(X) / df(X)
        http://python.jobbole.com/85295/
    '''
    iter_num = 0
    x_iter = x0
    while abs_error(f, x_iter) > precision:
        iter_num += 1
        x_iter = x_iter - f(x_iter) / df(x_iter)
        # print(iter_num, x_iter, df(x_iter), f(x_iter), abs_error(f, x_iter) > precision)
    return x_iter, iter_num 

if __name__ == '__main__':
    def f(x):
        return 6*x**5-5*x**4-4*x**3+3*x**2
    def df(x):
        return 30*x**4-20*x**3-12*x**2+6*x

    x, iter_num = newtons_iteration(f, df, 0.5, 1e-6)
    print(x, iter_num, f(x), df(x))
    x, iter_num = gradient_descent(f, df, 0.5, 0.25, 1e-6)