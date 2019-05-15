# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'KNN/'
    sys.path.append(__ML_PATH__)
    from Tuning.datatune import *
except ModuleNotFoundError:
    __F_PATH__ = os.getcwd() + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../')
    pass
__ALGO_PATH__ = __F_PATH__
sys.path.append(__ML_PATH__)

from Tuning.datatune import *
from Tuning.math import l2_distance
from sklearn.neighbors import KNeighborsClassifier as kNN

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
class knn_sklearn(object):
    @jit
    def __init__(self, ver_data, ver_labels, k):
        self._knn_kernel = kNN(n_neighbors=k, algorithm='auto')
        self._knn_kernel.fit(ver_data, ver_labels)
        return None

    @jit
    def predict(self, testcase):
        return self._knn_kernel.predict(testcase.reshape(1, -1))

class knn(object):
    @jit
    def __init__(self, ver_data, ver_labels, k):
        self._ver_num = ver_data.shape[0]
        self._ver_data = ver_data
        self._ver_labels = ver_labels
        self._K = k
        return None

    @jit
    def predict(self, testcase):
        # axis==0行相加,axis==1列相加
        euclidean_distance = l2_distance(testcase, self._ver_data, dim=1)
        topk_index = np.argsort(euclidean_distance, kind='quicksort')[:self._K]
        topk_label = [self._ver_labels[index] for index in topk_index]
        count_common = collect_np(topk_label)
        top1_label = get_topn(count_common, 1)
        return top1_label