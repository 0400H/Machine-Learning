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

def data_loader(dateset_dir) :
    labels = []
    file_list = os.listdir(dateset_dir)
    case_num = len(file_list)
    data_array = np.ndarray(shape=(case_num, 1024))

    for i in range(case_num):
        filename = file_list[i]
        classify_number = int(filename.split('_')[0])
        labels.append(classify_number)
        data_array[i] = img2col(dateset_dir + '/' + filename, 0, 32, 0, 32, np.int, 'utf-8')

    return data_array, labels

"""
Function description: kNN算法, 分类器
"""
class handwriting_sklearn(object):
    def __init__(self, ver_data, ver_labels, K) :
        self._knn_kernel = kNN(n_neighbors = K, algorithm = 'auto')
        self._knn_kernel.fit(ver_data, ver_labels)
        return None

    def classify(self, testcase) :
        return self._knn_kernel.predict(testcase.reshape(1, 1024))

class handwriting(object) :
    def __init__(self, ver_data, ver_labels, K) :
        self._ver_num = ver_data.shape[0]
        self._ver_data = ver_data
        self._ver_labels = ver_labels
        self._K = K
        return None

    def classify(self, testcase) :
        #create ndarray, shape = (ver_num, 1)
        test_ndarray = np.tile(testcase, (self._ver_num, 1)) - self._ver_data
        #平方后元素相加,axis==0列相加,axis==1行相加
        sq_distances = np.square(test_ndarray).sum(axis=1)
        #开方,计算出距离
        distances = np.power(sq_distances, 0.5)
        #返回distances中元素从小到大排序后的索引值
        sorted_distances = distances.argsort()
        #定一个记录类别次数的字典
        classCount = {}
        for i in range(self._K):
            #取出前k个元素的类别
            label = self._ver_labels[sorted_distances[i]]
            #计算类别次数， dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
            classCount[label] = classCount.get(label, 0) + 1
        #reverse==True降序排序字典, itemgetter(0) 与 itemgetter(1)分别根据字典的键和值进行排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        # print('label,times:', sortedClassCount)
        #返回次数最多的类别,即所要分类的类别
        return sortedClassCount[0][0]

"""
函数说明: 手写数字分类测试
"""
def classify_test(class_classifier):
    # verification, test
    ver_data, ver_labels = data_loader(__Father_Root__ + 'trainingDigits')
    test_data, test_labels = data_loader(__Father_Root__ + 'testDigits')

    classfier = class_classifier(ver_data, ver_labels, 3)

    errorCount = 0.0
    test_num = len(test_labels)
    for index in range(test_num) :
        testcase = test_data[index]
        label = test_labels[index]
        classify_result = classfier.classify(testcase)
        print("分类返回结果为%d\t真实结果为%d" % (classify_result, label))
        if (classify_result != label):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/test_num))

if __name__ == '__main__':
    classify_test(handwriting)
    classify_test(handwriting_sklearn)
