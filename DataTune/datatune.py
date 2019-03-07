# -*- coding: UTF-8 -*-

import numpy as np

"""
Function description: 从文件读取数据到ndarray
"""

def data2ndarray(filename, in_dtype, nop, interval) :
    return np.loadtxt(filename, dtype=in_dtype, comments=nop, delimiter=interval)

def file2ndarray(filename, interval, encode) :
    file2list = open(filename, 'r', encoding = encode).readlines()
    #针对有 BOM 的 UTF-8 文本，应该去掉BOM，否则后面会引发错误。
    file2list[0] = file2list[0].lstrip('\ufeff')
    #得到文件行数,列数, 创建合适的 NumPy 矩阵
    num_of_row = len(file2list)
    num_of_column = file2list[0].count(interval) + 1
    data_ndarray = np.zeros((num_of_row, num_of_column)).astype(np.str) 

    index = 0
    for line in file2list :
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line2list = line.strip().split(interval)
        data_ndarray[index,:] = line2list
        index += 1

    return data_ndarray

def get_ndarray_dim(data_ndarray, h_start, h_end, w_start, w_end, out_dtype=np.str) :
    if h_end <= h_start + 1 and w_end <= w_start + 1:
        data_ndarray = data_ndarray[h_start, w_start]
    elif h_end > h_start + 1 and w_end <= w_start + 1:
        data_ndarray = data_ndarray[h_start:h_end, w_start]
    elif h_end <= h_start + 1 and w_end > w_start + 1:
        data_ndarray = data_ndarray[h_start, w_start:w_end]
    elif h_end > h_start + 1 and w_end > w_start + 1:
        data_ndarray = data_ndarray[h_start:h_end, w_start:w_end]

    return data_ndarray.astype(out_dtype)

"""
Function description:对数据进行归一化
"""
def autoNorm(dataSet):
    #获得数据的极值和范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #创建shape为dataSet的ndarray
    normDataSet = np.zeros(np.shape(dataSet))
    #获取dataSet的行数
    row = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (row, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (row, 1))
    #返回归一化数据结果,数据范围,最小值

    return normDataSet, ranges, minVals