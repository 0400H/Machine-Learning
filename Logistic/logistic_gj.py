# -*- coding:UTF-8 -*-

import os
from sys import path
__Father_Root__ = os.path.dirname(os.path.abspath(__file__)) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../')
path.append(__Project_Root__)

from Tuning.datatune import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random


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
函数说明: sigmoid函数
"""
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
函数说明:梯度上升算法
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
    weights_array - 每次更新的回归系数
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n,1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles,n)
    return weights.getA(),weights_array                                    #将矩阵转换为数组，并返回

"""
函数说明:改进的随机梯度上升算法
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
    weights_array - 每次更新的回归系数
"""
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化
    weights_array = np.array([])                                            #存储每次更新的回归系数
    for j in range(numIter):                                            
        dataIndex = list(range(m))
        for i in range(m):            
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归系数
            weights_array = np.append(weights_array,weights,axis=0)         #添加回归系数到数组中
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    weights_array = weights_array.reshape(numIter*m,n)                         #改变维度
    return weights,weights_array                                             #返回

"""
函数说明:绘制回归系数与迭代次数的关系
Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2
"""
def plotWeights(weights_array1, weights_array2):
    #设置汉字格式
    fontfile = r"c:\windows\fonts\simsun.ttc"
    # fontfile = r"/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
    # fontfile = r"/usr/share/fonts/opentype/dejavu-sans-mono/DejaVuSansMono.ttf"
    font = FontProperties(fname=fontfile, size=5)

    fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(10,5))

    x1 = np.arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=10, weight='bold', color='black')  
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black') 
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black') 
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=10, weight='bold', color='black')  
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black') 


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=10, weight='bold', color='black')  
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black') 
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black') 
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=10, weight='bold', color='black')  
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black') 

    plt.show()        

if __name__ == '__main__':
    dataMat, labelMat = data_loader(__Father_Root__ + 'testSet.csv')
    weights1,weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2,weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)