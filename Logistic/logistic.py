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
import matplotlib.pyplot as plt
import numpy as np
from sympy import *

"""
函数说明: f(x) = -x^2 + 233x的导数
"""
def f_derivative(x):
    return -4 * x + 233

"""
函数说明: 加载数据
"""
def data_loader(filename):
    file_array = file2array2(filename, np.str, '\t', 'utf-8')
    data_array = np.ones(shape=(len(file_array), 3))
    data_array[:, 1:] = file_array[:, :-1].astype(np.float)
    label_array = file_array[:, -1].astype(np.float).reshape(-1)
    return data_array, label_array

"""
函数说明: 梯度上升法求简单凸函数极值(vector space)
"""
def gradient_ascent_vector(func_derivative, learn_rate=1e-2, precision=1e-8):
    # Draw func and func derivative image
    x_list = [num for num in range(-100, 100)]
    f_list = [4*num - 2*(num**2) for num in x_list]
    f_d_list = [f_derivative(num) for num in x_list]
    canvas, figure = plt.subplots(nrows=1, ncols=2,sharex=False, sharey=False, figsize=(8, 4))
    data2plt(figure[0], x_list, f_list, 'orange', 5, 0.5, u'', 9,
             'bold', 'red', u'x', 7, 'bold', 'black', u'f', 7, 'bold', 'black')
    data2plt(figure[1], x_list, f_d_list, 'orange', 5, 0.5, u'', 9,
             'bold', 'red', u'x', 7, 'bold', 'black', u'f\'', 7, 'bold', 'black')
    plt.show(canvas)

    # Iterative calculation of extremum
    x_old = 0
    x_new = x_old + 2 * precision
    num_iter = 0
    while abs(x_new - x_old) > precision :
        num_iter += 1
        x_old = x_new
        x_new = x_old + learn_rate * func_derivative(x_old)
    return x_new, num_iter

"""
函数说明: 梯度上升算法求解 w0 + w1 * x + w2 * y = 0 的权重矩阵
Parameters:
    data_ndarray - 数据集
    label_list - 数据标签
Returns:
    weights.getA() - 求得的权重矩阵(最优参数)
"""
def gradient_ascent_matrix(data_ndarray, label_list, learn_rate=1e-2, precision=1e-8):
    data_marix = np.matrix(data_ndarray)
    n = np.shape(data_marix)[1]
    weights = np.matrix(np.ones((n, 1)))
    label_matrix = np.matrix(label_list).reshape(-1, 1)
    avg_loss, num_iter = 2*precision, 0
    while (avg_loss >= precision):
        result = sigmoid(np.dot(data_marix, weights))
        loss = label_matrix - result
        avg_loss = abs(np.average(loss))
        weights += learn_rate * np.dot(data_marix.transpose(), loss)
        num_iter += 1
    return weights.getA(), num_iter                  # matrix to ndarray

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
            xcord1.append(data_matrix[i,1]);
            ycord1.append(data_matrix[i,2])
        else:
            xcord2.append(data_matrix[i,1]);
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
    extremum, num_iter = gradient_ascent_vector(f_derivative, 1e-1, 1e-8)
    print('extremum: %f, learn_rate: %.2e, precision: %.2e, iteration: %d' % (float(extremum), 1e-1, 1e-8, num_iter))

    data_matrix, label_matrix = data_loader(__F_PATH__ + 'testSet.csv')
    weights, num_iter = gradient_ascent_matrix(data_matrix, label_matrix, 1.2e-2, 1e-8)
    print('learn_rate: %.2e, precision: %.2e, iteration: %d' % (1.2e-2, 1e-8, num_iter))
    plotBestFit(weights)