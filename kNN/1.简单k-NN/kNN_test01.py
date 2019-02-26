# -*- coding: UTF-8 -*-
import numpy as np
import collections
# import operator

'''
    Function description:
        read dataset from csv
    Parameters:
        file
        start
        end
    Returns:
        data
        label
'''
def ReadDataSetCSV(file, start, end):
    csvdataset = np.loadtxt(file, dtype=str, comments='#', delimiter=',')
    data = csvdataset[start:end, 1:].astype(np.float)
    labels = csvdataset[start:end, 0]
    print(data)
    return data, labels

'''
    函数说明:kNN算法,分类器
    Parameters:
        testcase     - 用于分类的数据(测试集)
        dataSet - 用于训练的数据(训练集)
        labes   - 分类标签
        k       - kNN算法参数,选择距离最小的k个点
    Returns:
        sortedClassCount[0][0] - 分类结果
'''
def classify0(testcase, dataset, labels, k):
    # 计算欧式距离
    distances = np.sum((testcase - dataset)**2, axis=1)**0.5
    # k个最近数据的index
    k_data_index = np.argsort(distances, kind='quicksort')[0 : k]
    print(k_data_index)
    # k个最近的label
    k_labels = [labels[index] for index in k_data_index]
    print(k_labels)
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label

if __name__ == '__main__':
    date, labels = ReadDataSetCSV('knn.csv', 0, 6)
    #测试集
    case = [101, 20]
    #kNN分类
    test_class = classify0(case, date, labels, 3)
    #打印分类结果
    print(test_class)
