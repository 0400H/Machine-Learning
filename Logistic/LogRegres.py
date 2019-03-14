# -*- coding:UTF-8 -*-

import os
from sys import path
__Father_Root__ = os.path.dirname(__file__) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../')
path.append(__Project_Root__)

from DataTune.datatune import *
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:梯度上升算法测试函数
求函数f(x) = -x^2 + 4x的极大值
"""
def Gradient_Ascent_test():
    def f_prime(x_old):                                    #f(x)的导数
        return -2 * x_old + 4
    x_old = -1                                            #初始值，给一个小于x_new的值
    x_new = 0                                            #梯度上升算法初始值，即从(0,0)开始
    alpha = 0.01                                        #步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001                                #精度，也就是更新阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)            #上面提到的公式
    print(x_new)                                        #打印最终求解的极值近似值

"""
函数说明:加载数据
Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
"""

def data_loader(filename):
    file_array = file2array2(filename, np.str, '\t', 'utf-8')
    data_array = np.ones(shape=(len(file_array), 3))
    data_array[:, 1:] = file_array[:, :-1].astype(np.float)
    label_array = file_array[:, -1].astype(np.float).reshape(-1)

    return data_array, label_array

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
    dataMat, labelMat = data_loader(__Father_Root__ + 'testSet.csv')    
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)