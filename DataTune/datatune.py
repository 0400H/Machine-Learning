# -*- coding: UTF-8 -*-

import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

"""
Function description: 从文件读取数据到array
"""

def data2array(filename, in_dtype, nop, interval) :
    return np.loadtxt(filename, dtype=in_dtype, comments=nop, delimiter=interval)

def file2array(filename, interval, encode) :
    file2list = open(filename, 'r', encoding = encode).readlines()
    #针对有 BOM 的 UTF-8 文本，应该去掉BOM，否则后面会引发错误。
    file2list[0] = file2list[0].lstrip('\ufeff')
    #得到文件行数,列数, 创建合适的 NumPy 矩阵
    num_of_row = len(file2list)
    num_of_column = file2list[0].count(interval) + 1
    data_array = np.zeros((num_of_row, num_of_column)).astype(np.str) 

    index = 0
    for line in file2list :
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line2list = line.strip().split(interval)
        data_array[index,:] = line2list
        index += 1

    return data_array

def get_array_dim(data_array, h_start, h_end, w_start, w_end, out_dtype=np.str) :
    if h_end <= h_start + 1 and w_end <= w_start + 1:
        data_array = data_array[h_start, w_start]
    elif h_end > h_start + 1 and w_end <= w_start + 1:
        data_array = data_array[h_start:h_end, w_start]
    elif h_end <= h_start + 1 and w_end > w_start + 1:
        data_array = data_array[h_start, w_start:w_end]
    elif h_end > h_start + 1 and w_end > w_start + 1:
        data_array = data_array[h_start:h_end, w_start:w_end]

    return data_array.astype(out_dtype)

"""
Function description:对数据进行归一化
"""
def autoNorm(dataSet) :
    #获得数据的极值和范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #创建shape为dataSet的array
    normDataSet = np.zeros(np.shape(dataSet))
    #获取dataSet的行数
    row = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (row, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (row, 1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals

"""
Function description:
    可视化数据
Parameters:
    datingDataMat - 特征矩阵
    datingLabels  - 分类Label
Returns:
    none
"""

def get_marker_Line2D(ml_color, ml_markersize, ml_label) :
    return mlines.Line2D([], [], color=ml_color, marker='.', markersize=ml_markersize, label=ml_label)

def add_legend(plt_figure, handle_ml) :
    plt_figure.legend(handles=handle_ml)
    return None

def show_pyplot(plt) :
    return plt.show()

def data2plt(plt_figure, figure_name, data_x, data_y,
             font_file, color_dim_x, plt_color_list,
             point_pixel, point_transparency,
             t_label, t_fontsize, t_weight, t_color, 
             x_label, x_fontsize, x_weight, x_color,
             y_label, y_fontsize, y_weight, y_color) :
    plt_figure.scatter(x=data_x, y=data_y, color=plt_color_list,
                       s=point_pixel, alpha=point_transparency)
    plt_font = FontProperties(fname=font_file, size=14)
    title_text = plt_figure.set_title(t_label, FontProperties=plt_font)
    x_text = plt_figure.set_xlabel(x_label, FontProperties=plt_font)
    y_text = plt_figure.set_ylabel(y_label, FontProperties=plt_font)
    plt.setp(title_text, size=t_fontsize, weight=t_weight, color=t_color)
    plt.setp(x_text, size=x_fontsize, weight=x_weight, color=x_color)
    plt.setp(y_text, size=y_fontsize, weight=y_weight, color=y_color)
    return None