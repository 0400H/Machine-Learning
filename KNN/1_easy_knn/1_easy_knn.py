# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __FATHER_PATH__ = os.path.dirname(os.path.abspath(__file__)) + '/'
    __ML_PATH__ = os.path.abspath(__FATHER_PATH__ + '../../')
except NameError:
    try:
        __FATHER_PATH__ = os.getcwd() + '/'
        __ML_PATH__ = os.path.abspath(__FATHER_PATH__ + '../../')
        from DataTune.datatune import *
    except ModuleNotFoundError:
        __ML_PATH__ = os.getcwd() + '/'
        __FATHER_PATH__ = __ML_PATH__ + 'KNN/1_easy_knn/'
        pass
    pass
__ALGO_PATH__ = os.path.abspath(__ML_PATH__ + '/KNN')
sys.path.append(__ML_PATH__)
sys.path.append(__ALGO_PATH__)
print(__ML_PATH__, __ALGO_PATH__, __FATHER_PATH__, sep='\n')

from knn import *
from DataTune.datatune import *
from DataTune.logger import info

# %% Function description: read data_array, labels from csv
@jit
def data_loader(filename, nop, interval, batch_dim, feature_dim, label_dim):
    (h_start, h_end), (w_start, w_end) = batch_dim, feature_dim
    data_array = file2array1(filename, np.str, nop, interval)
    data = data2matrix(data_array, h_start, h_end, w_start, w_end, np.float)
    labels = data2col(data_array, h_start, h_end, label_dim, label_dim+1, np.str)
    return data, labels

'''
    函数说明: KNN分类器
    Parameters:
        testcase     - 用于分类的测试用例
        val_dataset  - 用于验证的数据
        labes        - 用于验证的标签
        k            - 选择距离最小的k个点
    Returns:
        top1_label   - 分类结果
'''
@jit
def classify_easy_knn(testcase, val_dataset, val_labels, k):
    # axis==0行相加,axis==1列相加
    l2_distance =  np.power(np.sum(np.square(testcase - val_dataset), axis = 1), 0.5)
    topk_index = np.argsort(l2_distance, kind='quicksort')[:k]
    topk_label = [val_labels[index] for index in topk_index]
    count_common = collect_np(topk_label)
    top1_label = get_topn(count_common, 1)
    return top1_label

#%%
if __name__ == '__main__':
    val_data, val_labels = data_loader(__FATHER_PATH__ + 'knn.csv', '#', ',', (0, 9), (1, 3), 0)
    test_dataset = np.array([[3, 11], [31, 42], [79, 80]])

    classfier_1 = knn(val_data, val_labels, 3)
    classfier_2 = knn_sklearn(val_data, val_labels, 3)

    for testcase in test_dataset:
        category_1 = classfier_1.classify(testcase)
        category_2 = classfier_2.classify(testcase)
        print(testcase, ': ', category_1, category_2[0])