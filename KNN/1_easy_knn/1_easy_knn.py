# -*- coding: UTF-8 -*-

import os
from sys import path
__Father_Root__ = os.path.dirname(__file__) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../../')
path.append(__Project_Root__)

print(__Father_Root__)
from DataTune.datatune import *

import numpy as np
import collections
# import operator

'''
    Function description:
        read data_array fpom csv
    Parameters:
        filename
        label_dim
        batch_dim
    Returns:
        data
        label
'''
def dataset2date_label(filename, nop, interval, batch_dim, feature_dim, label_dim) :
    (h_start, h_end), (w_start, w_end) = batch_dim, feature_dim
    data_array = data2array(filename, np.str, nop, interval)
    data = get_array_dim(data_array, h_start, h_end, w_start, w_end, np.float)
    labels = get_array_dim(data_array, h_start, h_end, label_dim, label_dim, np.str)

    return data, labels

'''
    函数说明:kNN算法,分类器
    Parameters:
        testcase - 用于分类的数据(测试集)
        dataSet  - 用于训练的数据(训练集)
        labes    - 分类标签
        k        - kNN算法参数,选择距离最小的k个点
    Returns:
        sortedClassCount[0][0] - 分类结果
'''
def classify1(testcase, dataset, labels, k):
    # 计算欧式距离
    distances = np.sum((testcase - dataset) ** 2, axis = 1) ** 0.5
    # k个最近数据的index
    k_index = np.argsort(distances, kind = 'quicksort')[0 : k]
    print(k_index)
    # k个最近的label
    k_labels = [labels[index] for index in k_index]
    print(k_labels)
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label

if __name__ == '__main__':
    date, labels = dataset2date_label(__Father_Root__ + 'knn.csv', '#', ',', (0, 6), (1, 3), 0)
    #测试集
    case = [101, 20]
    #kNN分类
    test_class = classify1(case, date, labels, 3)
    #打印分类结果
    print(test_class)
