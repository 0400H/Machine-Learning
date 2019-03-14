# -*- coding:UTF-8 -*-

import os
from sys import path
__Father_Root__ = os.path.dirname(os.path.abspath(__file__)) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../')
path.append(__Project_Root__)

from DataTune.datatune import *
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:加载数据
"""

def data_loader(filename):
    file_array = file2array2(filename, np.str, '\t', 'utf-8')
    data_array = np.ones(shape=(len(file_array), 3))
    data_array[:, 1:] = file_array[:, :-1].astype(np.float)
    label_array = file_array[:, -1].astype(np.float).reshape(-1)

    return data_array, label_array


def showdatas(x, f, f_p) :
    fontfile = r"c:\windows\fonts\simsun.ttc"
    # fontfile = r"/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
    # fontfile = r"/usr/share/fonts/opentype/dejavu-sans-mono/DejaVuSansMono.ttf"

    canvas, figure = plt.subplots(nrows=1, ncols=2,sharex=False, sharey=False, figsize=(16, 8))
    data2plt(figure[0], '00', x, f, fontfile, True, 'orange', 5, 0.5, u'', 9,
             'bold', 'red', u'x', 7, 'bold', 'black', u'f', 7, 'bold', 'black')
    data2plt(figure[1], '00', x, f_p, fontfile, True, 'orange', 5, 0.5, u'', 9,
             'bold', 'red', u'x', 7, 'bold', 'black', u'f\'', 7, 'bold', 'black')
    show_pyplot(plt)

"""
函数说明:梯度上升算法测试函数
求函数f(x) = -x^2 + 4x的极大值
"""
def f_derivative(x):                                  #f(x)的导数
    return -2 * x + 4

def Gradient_Ascent_test(func_derivative, lr=0.001):

    x_list = [num for num in range(-50, 50)]
    f_prime_list = [f_prime(num) for num in x_list]
    f_list = [4*num - 2*(num**2) for num in x_list]
    showdatas(x_list, f_list, f_prime_list)

    presision = 0.00000001                               #精度，也就是更新阈值
    x_old = 0                                            #初始值，给一个小于x_new的值
    x_new = x_old + 2 * presision                        #梯度上升算法初始值，即从(0,0)开始
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + lr * func_derivative(x_old)      #上面提到的公式
    print(x_new)                                         #打印最终求解的极值近似值

"""
函数说明:sigmoid函数
Parameters:
    inX - 数据
Returns:
    sigmoid函数
"""
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
函数说明:梯度上升算法

Parameters:
    data_ndarray - 数据集
    labels_list - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
"""
def gradAscent(data_ndarray, labels_list):
    dataMatrix = np.mat(data_ndarray)                                        #转换成numpy的mat
    labelMat = np.mat(labels_list).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                           #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                         #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                       #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                 #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                                                 #将矩阵转换为数组，返回权重数组

"""
函数说明:绘制数据集
"""
def plotDataSet():
    dataMat, labelMat = data_loader(__Father_Root__ + 'testSet.csv')                                     #加载数据集
    dataArr = np.array(dataMat)                                           #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                              #数据个数
    xcord1 = []; ycord1 = []                                              #正样本
    xcord2 = []; ycord2 = []                                              #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])      #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])      #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                             #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)  #绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)              #绘制负样本
    plt.title('DataSet')                                                  #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()                                                            #显示

"""
函数说明:绘制数据集
"""
def plotBestFit(weights):
    dataMat, labelMat = data_loader(__Father_Root__ + 'testSet.csv')                                     #加载数据集
    dataArr = np.array(dataMat)                                           #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                              #数据个数
    xcord1 = []; ycord1 = []                                              #正样本
    xcord2 = []; ycord2 = []                                              #负样本
    for i in range(n):                                                    #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])      #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])      #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                             #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)  #绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)              #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                  #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                    #绘制label
    plt.show()        

if __name__ == '__main__':
    # dataMat, labelMat = data_loader(__Father_Root__ + 'testSet.csv')    
    # weights = gradAscent(dataMat, labelMat)
    # plotBestFit(weights)
    Gradient_Ascent_test()