# -*- coding:UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Logistic/'
    sys.path.append(__ML_PATH__)
    from Tuning.datatune import *
except ModuleNotFoundError:
    __F_PATH__ = os.getcwd() + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../')
    pass
__ALGO_PATH__ = __F_PATH__
sys.path.append(__ML_PATH__)

from Tuning.math import *
from Tuning.datatune import *
from Tuning.logger import info
from sympy import *


class logistic(object):
    '''
    函数说明: 梯度上升算法求使
              L(w) = Y*W_T*X - Ln(1+e^(W_T*X))
              获取最大值的 W_T
    Parameters:
        data_array - 数据集 m*k
        label_list - 数据标签 m*1
    Returns:
        weights - 求得的权重矩阵(最优参数)
    '''
    @jit
    def fit(self, data_array, label_array, learn_rate=1e-2, precision=1e-8):
        label_array = label_array.reshape(-1, 1)
        feature_size = np.shape(data_array)[1]
        self.weight = np.ones(shape=(feature_size, 1), dtype=np.float)
        data_array_trans = data_array.transpose()
        df_vec = label_array - sigmoid(np.dot(data_array, self.weight))
        df_vec = np.dot(data_array.transpose(), df_vec)
        max_abs_df = np.max(np.abs(df_vec))

        # gradient boost method
        num_iter = 0
        while max_abs_df >= precision:
            df_vec = label_array - sigmoid(np.dot(data_array, self.weight))
            df_vec = np.dot(data_array_trans, df_vec)
            max_abs_df = np.max(np.abs(df_vec))

            for index in range(len(df_vec)):
                df = df_vec[index]
                if np.abs(df) >= precision:
                    self.weight[index] += learn_rate * df
                else:
                    continue
            num_iter += 1
            # info(num_iter, df_vec, self.weight)
        return self.weight, num_iter

    def predict(self, test_array):
        p1 = sigmoid(np.dot(test_array, self.weight).reshape(-1))
        p0 = 1 - p1
        result = np.zeros(len(test_array), dtype=np.int)
        result[p1 > p0] = 1
        return result