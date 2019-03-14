# -*- coding: UTF-8 -*-

import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

"""
Function description: 从文件读取数据到array
"""

def file2array1(filename, in_dtype, nop='#', interval='') :
    if (interval == '') :
        return np.loadtxt(filename, dtype=in_dtype, comments=nop)
    else :
        return np.loadtxt(filename, dtype=in_dtype, comments=nop, delimiter=interval)

def file2array2(filename, out_dtype=np.str, interval='', encode='utf-8') :
    file2list = open(filename, 'r', encoding = encode).readlines()
    #针对有 BOM 的 UTF-8 文本，应该去掉BOM，否则后面会引发错误。
    file2list[0] = file2list[0].lstrip('\ufeff')
    #得到文件行数,列数, 创建合适的 NumPy 矩阵
    data_num = len(file2list)
    num_of_column = len(file2list[0]) if interval == '' else file2list[0].count(interval) + 1
    data_array = np.zeros((data_num, num_of_column)).astype(out_dtype)

    index = 0
    for line in file2list :
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line2list = line.strip() if interval == '' else line.strip().split(interval)
        data_array[index] = np.array(line2list)
        index += 1

    return data_array.astype(out_dtype)

def str2col(string, out_dtype = np.int) :
    length = len(string.strip())
    str_col = np.zeros(shape = (length)).astype(out_dtype)
    for index in range(length) :
        str_col[index] = string[index]
    return str_col.astype(out_dtype).reshape(length)

def data2matrix(data_array, h_start, h_end, w_start, w_end, out_dtype=np.str) :
    return data_array[h_start:h_end, w_start:w_end].astype(out_dtype)

def data2col(data_array, h_start, h_end, w_start, w_end, out_dtype=np.str, out_row=True) :
    h_length = h_end - h_start
    w_length = w_end - w_start
    data_array_by_dim = data_array[h_start:h_end, w_start:w_end]
    if out_row == True :
        data_array_by_dim.flatten().astype(out_dtype)
    else :
        data_array_by_dim.flatten('F').astype(out_dtype)
    return data_array_by_dim.reshape(h_length * w_length)

def img2col(filename, h_start, h_end, w_start, w_end, out_dtype=np.int, encode='utf-8'):
    h_length = h_end - h_start
    w_length = w_end - w_start
    img_col = np.ndarray(shape=(h_length, w_length))
    fp_data = open(filename, 'r', encoding = encode).readlines()
    for h_index in range(h_start, h_end) :
        line_string = (fp_data[h_index])[w_start:w_end]
        img_col[h_index - h_start] = str2col(line_string, out_dtype)

    return img_col.reshape(h_length * w_length)

"""
Function description: 对数据进行归一化, 标准化
"""
def normalization(data_ndarray) :
    #获得数据的极值和范围
    minVals = data_ndarray.min(0)
    maxVals = data_ndarray.max(0)
    ranges = maxVals - minVals
    data_num = data_ndarray.shape[0]

    data_range = np.tile(ranges, (data_num, 1))
    data_loss = data_ndarray - np.tile(minVals, (data_num, 1))

    resault = data_loss / data_range
    return resault, ranges, minVals

def standardization(data_ndarray) :
    meanVals = data_ndarray.mean(0)
    data_num = data_ndarray.shape[0]

    data_mean = np.tile(meanVals, (data_num, 1))
    data_std = np.std(data_ndarray)

    resault = (data_ndarray - data_mean) / data_std
    return resault

"""
Function description: 可视化数据
"""

def get_marker_Line2D(ml_color, ml_markersize, ml_label) :
    return mlines.Line2D([], [], color=ml_color, marker='.', markersize=ml_markersize, label=ml_label)

def add_legend(plt_figure, handle_ml) :
    plt_figure.legend(handles=handle_ml)
    return None

def show_pyplot(plt) :
    return plt.show()

def data2plt(plt_figure, data_x, data_y,
             color_list, point_pixel, point_trans,
             t_label, t_fontsize, t_weight, t_color,
             x_label, x_fontsize, x_weight, x_color,
             y_label, y_fontsize, y_weight, y_color,
             fontfile_path = '') :
    if fontfile_path != '' :
        font = FontProperties(fname=fontfile_path, size=10)
        plt_figure.scatter(x=data_x, y=data_y, color=color_list,
                        s=point_pixel, alpha=point_trans)
        t_text = plt_figure.set_title(t_label, FontProperties=font)
        x_text = plt_figure.set_xlabel(x_label, FontProperties=font)
        y_text = plt_figure.set_ylabel(y_label, FontProperties=font)
        plt.setp(t_text, size=t_fontsize, weight=t_weight, color=t_color)
        plt.setp(x_text, size=x_fontsize, weight=x_weight, color=x_color)
        plt.setp(y_text, size=y_fontsize, weight=y_weight, color=y_color)
    else :
        plt_figure.scatter(x=data_x, y=data_y, color=color_list,
                        s=point_pixel, alpha=point_trans)
        t_text = plt_figure.set_title(t_label)
        x_text = plt_figure.set_xlabel(x_label)
        y_text = plt_figure.set_ylabel(y_label)
        plt.setp(t_text, size=t_fontsize, weight=t_weight, color=t_color)
        plt.setp(x_text, size=x_fontsize, weight=x_weight, color=x_color)
        plt.setp(y_text, size=y_fontsize, weight=y_weight, color=y_color)
    return None