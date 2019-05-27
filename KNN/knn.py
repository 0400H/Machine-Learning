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
        test_data     - 用于分类的测试用例
        val_dataset  - 用于验证的数据
        labes        - 用于验证的标签
        k            - 选择距离最小的k个点
    Returns:
        top1_label   - 分类结果
'''
class knn_sklearn(object):
    def __init__(self, k):
        self._knn_kernel = kNN(n_neighbors=k, algorithm='auto')
        return None

    def fit(self, data_array, label_array):
        self._knn_kernel.fit(data_array, label_array)
        return self

    def predict(self, test_data):
        return self._knn_kernel.predict(test_data.reshape(1, -1))

    def score(self, val_data, val_label):
        return self._knn_kernel.score(val_data, val_label)

class knn(object):
    def __init__(self, k):
        self._K = k
        return None

    def fit(self, data_array, label_array):
        self._num = data_array.shape[0]
        self._data = data_array
        self._label = label_array
        return self

    def predict(self, test_data):
        # axis==0行相加,axis==1列相加
        euclidean_distance = l2_distance(test_data, self._data, dim=1)
        topk_index = np.argsort(euclidean_distance, kind='quicksort')[:self._K]
        topk_label = [self._label[index] for index in topk_index]
        count_common = collect_np(topk_label)
        top1_label = get_topn(count_common, 1)
        return top1_label

    def score(self, val_data, val_label):
        errorCount = 0.0
        val_num = len(val_label)
        for index in range(val_num):
            data = val_data[index]
            label = val_label[index]
            result = self.predict(data)
            # info("分类返回结果为%d, 真实结果为%d" % (predict_result, label))
            if (result != label):
                errorCount += 1.0
        # info("总共错了%d个数据, 错误率为%f%%" % (errorCount, errorCount/test_num))
        return 1 - errorCount/val_num