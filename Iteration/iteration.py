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

from Tuning.logger import info
from Tuning.datatune import *
from Tuning.math import *
from numba import jit
import numpy as np
import sympy

class iteration(object):
    def s_init(self, s_func, *s_args):
        self.s_args = s_args
        self.s_func = s_func
        self.s_dfunc = [self.s_df(arg) for arg in self.s_args]
        self.l_args = len(s_args)

    @jit
    def s_df(self, s_arg):
        return sympy.diff(self.s_func, s_arg)

    @jit
    def v_func(self, *v_args):
        args_map = dict(zip(self.s_args, v_args))
        return self.s_func.evalf(subs=args_map)

    @jit
    def v_dfunc(self, s_dfunc, *v_args):
        args_map = dict(zip(self.s_args, v_args))
        return s_dfunc.evalf(subs=args_map)

    @jit
    def v_f(self, v_args):
        args_map = dict(zip(self.s_args, v_args))
        return self.s_func.evalf(subs=args_map)

    @jit
    def v_df(self, v_args):
        args_map = dict(zip(self.s_args, v_args))
        v_df_array = np.zeros(shape=(self.l_args), dtype=np.float)
        for index in range(self.l_args):
            v_df_array[index] = self.s_dfunc[index].evalf(subs=args_map)
        return v_df_array

    @jit
    def gradient(self, mode, learn_rate, precision, *args0):
        iter_num = 0
        v_args = np.array(args0, dtype=np.float)
        mini_diff = self.v_df(v_args)
        while np.max(np.abs(mini_diff)) > precision:
            iter_num += 1
            mini_diff = self.v_df(v_args)
            for index in range(self.l_args):
                arg_mini_diff = mini_diff[index]
                if np.abs(arg_mini_diff) > precision:
                    if mode == 'descent':
                        v_args[index] = v_args[index] - learn_rate * arg_mini_diff
                    elif mode == 'boosting':
                        v_args[index] = v_args[index] + learn_rate * arg_mini_diff
                    else:
                        info('wrong mode !')
                        return None
                else:
                    continue
            # info(iter_num, v_args, self.v_df(v_args), self.v_f(v_args))
        return v_args, iter_num

    @jit
    def newtons_root(self, precision, *args0):
        iter_num = 0
        v_args = np.array(args0, dtype=np.float)
        while abs_error(self.v_f, v_args) > precision:
            iter_num += 1
            v_args = v_args - self.v_f(v_args) / self.v_df(v_args)
            # info(iter_num, v_args, self.v_df(v_args), self.v_f(v_args))
        return v_args, iter_num

    pass