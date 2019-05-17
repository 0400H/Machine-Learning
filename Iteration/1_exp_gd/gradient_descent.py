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
    # 6*x^5-5*x^4-4*x^3+3*x^2
    s_f = 6*s_x**5-5*s_x**4-4*s_x**3+3*s_x**2
    return s_f, s_x

def func2():
    s_x, s_y = sympy.symbols('x y')
    # 2*(x+3)^2 + 4*(y-5)^2 + 6
    s_f = 2*(s_x+3)**2 + 4*(s_y-5)**2 + 6
    return s_f, s_x, s_y

def func3():
    s_x, s_y = sympy.symbols('x y')
    s_f = s_x**4 - 8*s_x*s_y + 2*s_y**2 - 3
    return s_f, s_x, s_y

if __name__ == '__main__':
    s_c = iteration()

    s_f, *s_args = func1()
    s_c.s_init(s_f, *s_args)
    info(s_c.newtons_root(1e-10, -100))
    info(s_c.newtons_root(1e-10, -0.5))
    info(s_c.newtons_root(1e-10, 100))
    info(s_c.gradient('descent', 1e-2, 2e-8, 0.5))
    info(s_c.gradient('descent', 1e-1, 1e-8, 0.1))
    info(s_c.gradient('boosting', 1e-2, 1e-8, -0.5))
    info(s_c.gradient('boosting', 1e-1, 1e-8, 0.1))

    s_f, *s_args = func2()
    s_c.s_init(s_f, *s_args)
    # info(s_c.newtons_root(1e-8, -10, 10))
    info(s_c.gradient('descent', 1e-1, 1e-8, -100, -100))

    s_f, *s_args = func3()
    s_c.s_init(s_f, *s_args)
    # info(s_c.newtons_root(1e-8, 1, 10))
    info(s_c.gradient('descent', 5e-3, 1e-8, 10, 10))