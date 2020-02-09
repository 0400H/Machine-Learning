#-*- coding: UTF-8 -*-

import operator
import collections
from numba import jit
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

"""
Function description: 从文件读取数据到array
"""
@jit
def File2ListArray(file_path='', jump_out_key=[], enter_key=[], select_key=[], dtype=np.float):
    '''
    param:
        jump_out_key: find any key, and drop the line
        enter_key: find all key, and select the line
        select_key: any tuple member of it will select data
    return:
        a array list, lenth of it is same with the select_key.
    '''
    select_key_length = len(select_key)
    data_list = [[] for i in range(select_key_length)]

    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            correct_line = True
            if type(jump_out_key) is type([]):
                for key in jump_out_key:
                    if key != '' and line.find(key) != -1:
                        correct_line = False
                        break
            else:
                if jump_out_key != '' and line.find(jump_out_key) != -1:
                    correct_line = False

            if correct_line == False:
                continue

            if enter_key != [] and enter_key != '':
                if type(enter_key) is type([]):
                    for key in enter_key:
                        if line.find(key) == -1:
                            correct_line = False
                            break
                elif type(enter_key) is type(''):
                    if line.find(enter_key) == -1:
                        correct_line = False

            if correct_line == False:
                continue

            for select_key_index in range(select_key_length):
                start_key, end_key = select_key[select_key_index]
                if end_key == '':
                    print('wrong end_key!')
                    break
                else:
                    if line.find(end_key) != -1:
                        start_index = 0
                        if start_key != '':
                            if line.find(start_key) != -1:
                                start_index = line.index(start_key) + len(start_key)
                            else:
                                continue
                        end_index = line.index(end_key)
                        string = line[start_index: end_index].strip('\n')
                        data_list[select_key_index].append(string)

    data_list_array, length_list = [], []
    for data in data_list:
        length_list.append(len(data))
        data_list_array.append(np.array(data).astype(dtype))
    return data_list_array, length_list

# @jit
def file2array1(file_path, in_dtype, nop='#', interval='', usecols=None, unpack=False):
    if (interval == ''):
        return np.loadtxt(file_path, dtype=in_dtype, comments=nop)
    else:
        return np.loadtxt(file_path, dtype=in_dtype, comments=nop, delimiter=interval)
    pass

# @jit
def file2array2(file_path, out_dtype=np.str, interval='', encode='utf-8'):
    file2list = open(file_path, 'r', encoding=encode).readlines()
    # 针对有 BOM 的 UTF-8 文本，应该去掉BOM，否则后面会引发错误。
    file2list[0] = file2list[0].lstrip('\ufeff')
    # 得到文件行数,列数, 创建合适的 NumPy 矩阵
    data_num = len(file2list)
    num_of_column = (len(file2list[0]), file2list[0].count(interval) + 1)[interval != '']
    data_array = np.zeros((data_num, num_of_column)).astype(out_dtype)

    index = 0
    for line in file2list :
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        line2list = (line, line.split(interval))[interval != '']
        data_array[index] = np.array(line2list)
        index += 1
    return data_array.astype(out_dtype)

@jit
def file2set(filename):
    words_set = set()
    fp = open(filename, 'r', encoding = 'utf-8')
    list_list = fp.readlines()
    for line in list_list:
        line = line.strip()
        if len(line) > 0:
            words_set = words_set | set(line)
        else:
            continue
    return words_set

@jit
def img2col(file_path, h_start, h_end, w_start, w_end, out_dtype=np.int, encode='utf-8'):
    h_length = h_end - h_start
    w_length = w_end - w_start
    img_col = np.ndarray(shape=(h_length, w_length))
    fp_data = open(file_path, 'r', encoding = encode).readlines()
    for h_index in range(h_start, h_end):
        line_string = (fp_data[h_index])[w_start:w_end]
        img_col[h_index - h_start] = str2col(line_string, out_dtype)
    return img_col.reshape(h_length * w_length)

@jit
def str2col(string, out_dtype = np.int):
    length = len(string.strip())
    str_col = np.zeros(shape = (length)).astype(out_dtype)
    for index in range(length):
        str_col[index] = string[index]
    return str_col.astype(out_dtype).reshape(length)

def data2matrix(data_array, h_start, h_end, w_start, w_end, out_dtype=np.str):
    return data_array[h_start:h_end, w_start:w_end].astype(out_dtype)

def data2col(data_array, h_start, h_end, w_start, w_end, out_dtype=np.str, out_row=True):
    h_length = h_end - h_start
    w_length = w_end - w_start
    data_array_by_dim = data_array[h_start:h_end, w_start:w_end]
    if out_row == True :
        data_array_by_dim.flatten().astype(out_dtype)
    else:
        data_array_by_dim.flatten('F').astype(out_dtype)
    return data_array_by_dim.reshape(h_length * w_length)

"""
Function description: 数据处理
"""
def collect_std(data_list=[]):
    return dict(collections.Counter(data_list))

def collect_np(data_list=[]):
    return dict(zip(*np.unique(data_list, return_counts=True)))

def normalization(data_ndarray):
    # 获得列数据的极值和范围
    minVals = data_ndarray.min(axis=0)
    maxVals = data_ndarray.max(axis=0)
    ranges = maxVals - minVals
    line_num = data_ndarray.shape[0]

    data_range = np.tile(ranges, (line_num, 1))
    data_min = np.tile(minVals, (line_num, 1))
    data_loss = data_ndarray - data_min

    result = data_loss / data_range
    return result, ranges, minVals

def standardization(data_ndarray):
    meanVals = data_ndarray.mean(0)
    data_num = data_ndarray.shape[0]

    data_mean = np.tile(meanVals, (data_num, 1))
    data_std = np.std(data_ndarray)

    resault = (data_ndarray - data_mean) / data_std
    return resault

@jit
def outlier_process(data_array, length, gamma):
    data_mean = np.mean(data_array[:length])
    data_std = np.std(data_array[:length])
    for index in range(1, length):
        if data_array[index] > data_mean + gamma * data_std or \
           data_array[index] < data_mean - gamma * data_std :
            data_array[index] = data_array[index-1]
    return None

def auto_fit(data_x, data_y, start, end, degree):
    # Polynomial fitting
    coefs = poly.polyfit(data_x[start:end], data_y[start:end], degree)
    data_fit = poly.polyval(data_x[start:end], coefs)
    data_fit[data_fit < 0.0] = 0.0
    data_fit[data_fit > 1.0] = 1.0
    return data_fit

def get_topn(data_dict, n, mode='reduce'):
    topn = None
    if n == 1:
        if mode == 'reduce':
            topn = max(data_dict, key=data_dict.get)
        else:
            topn = min(data_dict, key=data_dict.get)
    else:
        if mode == 'reduce':
            topn = sorted(data_dict.items(), key=operator.itemgetter(1), reverse=True)[:n]
        else:
            topn = sorted(data_dict.items(), key=operator.itemgetter(1), reverse=False)[:n]
    return topn

def get_topn_case(data_array, n, mode='reduce'):
    case_index = None
    if mode == 'reduce':
        topn_case_index = np.argsort(data_array)[-n:]
    else:
        topn_case_index = np.argsort(data_array)[:n]
    topn_case = data_array[sorted(case_index)]
    return topn_case, topn_case_index


"""
Function description: 可视化数据
"""
def plt_add_marker_line2d(ml_label, ml_color, ml_markersize=5):
    return mlines.Line2D([], [], color=ml_color, marker='.', markersize=ml_markersize, label=ml_label)

def plt_add_title(plt_figure, t_title, t_fontsize=9, t_weight='bold', t_color='red'):
    t_text = plt_figure.set_title(t_title)
    plt.setp(t_text, size=t_fontsize, weight=t_weight, color=t_color)
    return None

def plt_add_xy_title(plt_figure, x_title=u'', y_title=u'',
                     x_fontsize=10, x_weight='bold', x_color='black',
                     y_fontsize=10, y_weight='bold', y_color='black'):
    x_text = plt_figure.set_xlabel(x_title)
    y_text = plt_figure.set_ylabel(y_title)
    plt.setp(x_text, size=x_fontsize, weight=x_weight, color=x_color)
    plt.setp(y_text, size=y_fontsize, weight=y_weight, color=y_color)
    return None

def data2plt(plt_figure, data_x, data_y,
             pixel_color, point_pixel, point_trans,
             t_title, t_fontsize, t_weight, t_color,
             x_title, x_fontsize, x_weight, x_color,
             y_title, y_fontsize, y_weight, y_color):
    plt_figure.scatter(x=data_x, y=data_y, color=pixel_color,
                    s=point_pixel, alpha=point_trans)
    t_text = plt_figure.set_title(t_title)
    x_text = plt_figure.set_xlabel(x_title)
    y_text = plt_figure.set_ylabel(y_title)
    plt.setp(t_text, size=t_fontsize, weight=t_weight, color=t_color)
    plt.setp(x_text, size=x_fontsize, weight=x_weight, color=x_color)
    plt.setp(y_text, size=y_fontsize, weight=y_weight, color=y_color)
    return None

def plt_draw2d(plt_figure, data_x, data_y, pixel_color, plt_type='mix',
               l_label=u'', t_title=u'', x_title=u'', y_title=u'',
               point_pixel=1, point_trans=0.5, grid=False, legend='right', 
               t_fontsize=10, t_weight='bold', t_color='red',
               x_fontsize=10, x_weight='bold', x_color='black',
               y_fontsize=10, y_weight='bold', y_color='black'):
    if plt_type == 'scatter':
        plt_figure.scatter(x=data_x, y=data_y, color=pixel_color, s=point_pixel, alpha=point_trans)
    elif type(pixel_color) == type(''):
        if plt_type == 'line':
            plt_figure.plot(data_x, data_y, pixel_color, label=l_label)
        elif plt_type == 'mix':
            plt_figure.scatter(x=data_x, y=data_y, color='orange', s=3*point_pixel, alpha=1.0)
            plt_figure.plot(data_x, data_y, pixel_color, label=l_label)
    else:
        print('algo line only support string for pixel_color')

    plt_add_title(plt_figure, t_title, t_fontsize, t_weight, t_color)
    plt_add_xy_title(plt_figure, x_title, y_title,
                     x_fontsize, x_weight, x_color,
                     y_fontsize, y_weight, y_color)

    plt_figure.grid(grid)
    if legend != 'none' and l_label != u'' and l_label != [] :
        if plt_type == 'scatter' and type(l_label) == type([]) and len(l_label[0]) == len(l_label[1]):
            l_handle = []
            length = len(l_label[0])
            for index in range(length):
                l_handle.append(plt_add_marker_line2d(l_label[0][index], l_label[1][index], t_fontsize))
            plt_figure.legend(handles=l_handle)
        if (plt_type == 'line' or plt_type == 'mix') and type(l_label) == type(''):
            plt_figure.legend(loc='right')
    return None