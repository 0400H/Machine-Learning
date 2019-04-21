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
        __FATHER_PATH__ = __ML_PATH__ + 'KNN/3_knn/'
        pass
    pass
sys.path.append(__ML_PATH__)
print(__ML_PATH__, __FATHER_PATH__, sep='\n')

from DataTune.datatune import *
from DataTune.logger import info
from sklearn.neighbors import KNeighborsClassifier as kNN

@jit
def data_loader(dateset_dir) :
    labels = []
    file_list = os.listdir(dateset_dir)
    case_num = len(file_list)
    data_array = np.ndarray(shape=(case_num, 1024))

    for i in range(case_num):
        filename = file_list[i]
        classify_number = int(filename.split('_')[0])
        labels.append(classify_number)
        data_array[i] = img2col(dateset_dir + '/' + filename, 0, 32, 0, 32, np.int, 'utf-8')

    return data_array, labels

"""
Function description: kNN算法, 分类器
"""
class knn_sklearn(object):
    @jit
    def __init__(self, ver_data, ver_labels, K) :
        self._knn_kernel = kNN(n_neighbors = K, algorithm = 'auto')
        self._knn_kernel.fit(ver_data, ver_labels)
        return None

    @jit
    def classify(self, testcase) :
        return self._knn_kernel.predict(testcase.reshape(1, 1024))

class knn(object) :
    def __init__(self, ver_data, ver_labels, K) :
        self._ver_num = ver_data.shape[0]
        self._ver_data = ver_data
        self._ver_labels = ver_labels
        self._K = K
        return None

    @jit
    def classify(self, testcase) :
        l2_distance =  np.power(np.sum(np.square(testcase - self._ver_data), axis = 1), 0.5)
        topk_index = np.argsort(l2_distance, kind='quicksort')[:self._K]
        topk_label = [self._ver_labels[index] for index in topk_index]
        count_common = collect_np(topk_label)
        top1_label = get_topn(count_common, 1)
        return top1_label

"""
函数说明: 手写数字分类测试
"""
@jit
def classify_test(class_classifier):
    # verification, test
    ver_data, ver_labels = data_loader(__FATHER_PATH__ + 'trainingDigits')
    test_data, test_labels = data_loader(__FATHER_PATH__ + 'testDigits')

    classfier = class_classifier(ver_data, ver_labels, 3)

    errorCount = 0.0
    test_num = len(test_labels)
    for index in range(test_num) :
        testcase = test_data[index]
        label = test_labels[index]
        classify_result = classfier.classify(testcase)
        info("分类返回结果为%d\t真实结果为%d" % (classify_result, label))
        if (classify_result != label):
            errorCount += 1.0
    info("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/test_num))

if __name__ == '__main__':
    classify_test(knn)
    classify_test(knn_sklearn)