# -*- coding: UTF-8 -*-

import os
from sys import path
__Father_Root__ = os.path.dirname(__file__) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../../')
path.append(__Project_Root__)

from DataTune.datatune import *

import numpy as np
import operator
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
函数说明: 将32x32的二进制图像转换为1x1024向量。
"""
def img2vector(filename):
    return img2col(filename, 0, 32, 0, 32, np.int, 'utf-8')

"""
Function description: kNN算法, 分类器
Parameters:
    inX     - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes   - 分类标签
    k       - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classify_handwriting(inX, dataSet, labels, K):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX， 共1次(横向), 行向量方向上重复inX， 共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #平方后元素相加,axis==0列相加,axis==1行相加
    sqDistances = np.square(diffMat).sum(axis=1)
    #开方,计算出距离
    distances = np.power(sqDistances, 0.5)
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(K):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #计算类别次数， dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #reverse==True降序排序字典, itemgetter(0) 与 itemgetter(1)分别根据字典的键和值进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print('label,times:', sortedClassCount)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

"""
函数说明: 手写数字分类测试
"""
def classify_test():
    #测试集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = os.listdir(__Father_Root__ + 'trainingDigits')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector(__Father_Root__ + 'trainingDigits/%s' % (fileNameStr))
    #返回testDigits目录下的文件名
    testFileList = os.listdir(__Father_Root__ + 'testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector(__Father_Root__ + 'testDigits/%s' % (fileNameStr))
        #获得预测结果
        classifierResult = classify_handwriting(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))

def classify_test_sklearn():
    #测试集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = os.listdir(__Father_Root__ + 'trainingDigits')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector(__Father_Root__ + 'trainingDigits/%s' % (fileNameStr))
    #构建kNN分类器
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    #拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = os.listdir(__Father_Root__ + 'testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector(__Father_Root__ + 'testDigits/%s' % (fileNameStr))
        #获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))

if __name__ == '__main__':
    classify_test()
    classify_test_sklearn()
