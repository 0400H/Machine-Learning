# -*- coding: UTF-8 -*-

import os, sys
__Father_Root__ = os.path.dirname(os.path.abspath(__file__)) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../../')
sys.path.append(__Project_Root__)

from DataTune.datatune import *
import numpy as np
import collections

'''
    Function description: read data_array, labels from csv
'''
def data_loader(filename, nop, interval, batch_dim, feature_dim, label_dim) :
    (h_start, h_end), (w_start, w_end) = batch_dim, feature_dim
    data_array = file2array1(filename, np.str, nop, interval)
    data = data2matrix(data_array, h_start, h_end, w_start, w_end, np.float)
    labels = data2col(data_array, h_start, h_end, label_dim, label_dim+1, np.str)

    return data, labels

'''
    函数说明:kNN算法,分类器
    Parameters:
        testcase     - 用于分类的数据(测试集)
        val_dataset  - 用于训练的数据(训练集)
        labes        - 分类标签
        k            - kNN算法参数,选择距离最小的k个点
    Returns:
        sortedClassCount[0][0] - 分类结果
'''
def classify_easy_knn(testcase, val_dataset, val_labels, k):
    # 计算欧式距离
    distances =  np.power(np.sum(np.square(testcase - val_dataset), axis = 1), 0.5)
    # k个最近数据的label， index
    k_index = np.argsort(distances, kind = 'quicksort')[0 : k]
    k_label = [str(val_labels[index]) for index in k_index]
    # print(k_index)
    # print(k_label)
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_label).most_common(1)[0][0]
    return label

if __name__ == '__main__':
    val_data, val_labels = data_loader(__Father_Root__ + 'knn.csv', '#', ',', (0, 9), (1, 3), 0)
    #测试集
    test_dataset = [(20, 90), (50, 60), (80, 10)]
    #kNN分类
    for testcase in test_dataset :
        classify_class = classify_easy_knn(testcase, val_data, val_labels, 9)
        print(testcase, ': ', classify_class)