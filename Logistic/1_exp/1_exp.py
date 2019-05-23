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
from sympy import *

def data_loader(filename):
    file_array = file2array2(filename, np.str, '\t', 'utf-8')
    label_array = file_array[:, -1].reshape(-1).astype(np.float)
    data_array = np.ones(shape=(len(file_array), 3), dtype=np.float)
    # X_i {1, x1, x2} W_i {b, w1, w2}
    data_array[:, 1:] = file_array[:, :-1].astype(np.float)
    return data_array, label_array

"""
函数说明: 梯度上升算法求解使
         L(w) = Y*W_T*X - Ln(1+e^(W_T*X))
         获取最大值的 W_T
Parameters:
    data_array - 数据集 N*K
    label_list - 数据标签 N*K
Returns:
    weights.getA() - 求得的权重矩阵(最优参数)
"""
def logistic_gradient_ascent(data_array, label_list, learn_rate=1e-2, precision=1e-8):
    data_array = data_array
    feature_size = np.shape(data_array)[1]
    weights = np.ones(shape=(feature_size, 1), dtype=np.float)
    label_matrix = np.array(label_list).reshape(-1, 1)
    df = label_matrix - sigmoid(np.dot(data_array, weights))
    df = np.dot(data_array.transpose(), df)
    abs_df = np.abs(np.sum(df))

    num_iter = 0
    while abs_df >= precision:
        df = label_matrix - sigmoid(np.dot(data_array, weights))
        df = np.dot(data_array.transpose(), df)
        abs_df = np.abs(np.average(df))
        weights += learn_rate * df
        num_iter += 1
        print(num_iter, abs_df)
    return weights, num_iter

"""
函数说明:绘制数据集
"""
def plotBestFit(weights):
    data_matrix, label_matrix = data_loader(__F_PATH__ + 'testSet.csv')
    num = np.shape(data_matrix)[0]
    xcord1, ycord1 = [], []                                               #正样本
    xcord2, ycord2 = [], []                                               #负样本
    for i in range(num):                                                  #根据数据集标签进行分类
        if int(label_matrix[i]) == 1:
            xcord1.append(data_matrix[i,1])
            ycord1.append(data_matrix[i,2])
        else:
            xcord2.append(data_matrix[i,1])
            ycord2.append(data_matrix[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', marker = 's', alpha=.5)
    # w0 + w1 * x + w2 * y = 0
    x = np.arange(-3.0, 3.0, 0.1)
    y = -(weights[0] + weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    data_matrix, label_matrix = data_loader(__F_PATH__ + 'testSet.csv')
    weights, num_iter = logistic_gradient_ascent(data_matrix, label_matrix, 1.2e-2, 1e-8)
    print('learn_rate: %.2e, precision: %.2e, iteration: %d' % (1.2e-2, 1e-8, num_iter))
    plotBestFit(weights)