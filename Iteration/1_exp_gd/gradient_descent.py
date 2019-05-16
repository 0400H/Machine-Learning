#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Iteration/1_exp_gd/'
    sys.path.append(__ML_PATH__)
    from Tuning.datatune import *
except ModuleNotFoundError:
    __F_PATH__ = os.getcwd() + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../../')
    pass
__ALGO_PATH__ = os.path.abspath(__F_PATH__ + '../')
sys.path.append(__ML_PATH__)
sys.path.append(__ALGO_PATH__)
print(__ML_PATH__, __ALGO_PATH__, __F_PATH__, sep='\n')

from Tuning.logger import info
from iteration import *

def func1():
    s_x = sympy.Symbol('x')
    s_f = 6*s_x**5-5*s_x**4-4*s_x**3+3*s_x**2
    return s_f, s_x

def func2():
    s_x, s_y = sympy.symbols('x y')
    s_f = s_x**2 + s_y
    return s_f, s_x, s_y

if __name__ == '__main__':
    s_c = symbol_compute()

    # 导数
    s_f, *s_args = func1()
    s_c.s_init(s_f, *s_args)
    v_fx = s_c.v_func(10)
    s_dfx = s_c.s_dfunc(s_c.s_args[0])
    v_dfx = s_c.v_dfunc(s_dfx, s_c.s_args[0], 10)
    info_format = '{}, {}, {}, {}'
    info(info_format.format(s_f, s_dfx, v_fx, v_dfx))

    # 偏导数
    s_f, *s_args = func2()
    s_c.s_init(s_f, *s_args)
    v_f = s_c.v_func(10, 1)
    s_dfx = s_c.s_dfunc(s_c.s_args[0])
    v_dfx = s_c.v_dfunc(s_dfx, s_c.s_args[0], 10)
    s_dfy = s_c.s_dfunc(s_c.s_args[1])
    v_dfy = s_c.v_dfunc(s_dfy, s_c.s_args[1], 10)
    info_format = '{}, {}, {}, {}, {}, {}'
    info(info_format.format(s_f, s_dfx, s_dfy, v_f, v_dfx, v_dfy))