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
from sklearn.linear_model import LogisticRegression

class logistic_sklearn(object):
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
    def __init__(self, max_iteration=10000):
        self._logistic_regression_kernel = LogisticRegression(solver='sag', max_iter=max_iteration)
        return None

    def fit(self, data_array, label_array):
        data_length, feature_size = np.shape(data_array)
        self.weight = np.zeros(shape=(feature_size), dtype=np.float)
        self._logistic_regression_kernel.fit(data_array, label_array, self.weight)
        return self

    def predict(self, test_data):
        test_result = self._logistic_regression_kernel.predict_proba(test_data)
        return test_result

    def score(self, val_data, val_label):
        val_score = self._logistic_regression_kernel.score(val_data, val_label)
        return val_score
        

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
    def fit(self, data_array, label_array, learn_rate=1e-2, precision=1e-8, method='sgd', history=False):
        label_array = label_array.reshape(-1, 1)
        data_length, feature_size = np.shape(data_array)
        self.weight_history = np.zeros(shape=(feature_size, 1), dtype=np.float)
        self.weight = np.zeros(shape=(feature_size, 1), dtype=np.float)
        if history == True:
            self.weight_history = self.weight.copy()

        num_iter = 0
        max_abs_df = precision
        current_lr = learn_rate
        if method == 'sgb':
            # stochastic gradient boost method
            index_array = np.arange(data_length)
            while max_abs_df >= precision:
                index_array_shuffle = np.random.permutation(index_array)
                data_array_shuffle = data_array[index_array_shuffle]
                label_array_shuffle = label_array[index_array_shuffle]
                for index in range(data_length):
                    rand_data = data_array_shuffle[index]
                    rand_data_trans = rand_data.transpose()
                    loss = label_array_shuffle[index] - sigmoid(np.dot(rand_data, self.weight))
                    df_vec = rand_data_trans * loss
                    max_abs_df = np.max(np.abs(df_vec))

                    for idx in range(len(df_vec)):
                        df = df_vec[idx]
                        if np.abs(df) >= precision:
                            self.weight[idx] += current_lr * df
                        else:
                            continue

                    if history == True:
                        self.weight_history = np.append(self.weight_history, self.weight.copy())
                    num_iter += 1
                    # info(num_iter, current_lr, max_abs_df, df_vec.reshape(-1), self.weight.reshape(-1))
        else :
            # batch gradient boost method
            while max_abs_df >= precision:
                data_array_trans = data_array.transpose()
                loss_vec = label_array - sigmoid(np.dot(data_array, self.weight))
                df_vec = np.dot(data_array_trans, loss_vec)
                max_abs_df = np.max(np.abs(df_vec))

                for index in range(len(df_vec)):
                    df = df_vec[index]
                    if np.abs(df) >= precision:
                        self.weight[index] += learn_rate * df
                    else:
                        continue

                if history == True:
                    self.weight_history = np.append(self.weight_history, self.weight.copy())
                num_iter += 1
                # info(num_iter, df_vec, self.weight)

        self.weight = self.weight.reshape(-1)
        self.weight_history = np.array(self.weight_history).reshape(-1, feature_size)
        return self.weight, num_iter

    def predict(self, test_data):
        p1 = sigmoid(np.dot(test_data, self.weight).reshape(-1))
        p0 = 1 - p1
        result = np.zeros(len(test_data), dtype=np.int)
        result[p1 > p0] = 1
        return result