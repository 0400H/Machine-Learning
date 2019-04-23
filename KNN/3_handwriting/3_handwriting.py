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
        from Tuning.datatune import *
    except ModuleNotFoundError:
        __ML_PATH__ = os.getcwd() + '/'
        __FATHER_PATH__ = __ML_PATH__ + 'KNN/3_knn/'
        pass
    pass
__ALGO_PATH__ = os.path.abspath(__ML_PATH__ + '/KNN')
sys.path.append(__ML_PATH__)
sys.path.append(__ALGO_PATH__)
print(__ML_PATH__, __ALGO_PATH__, __FATHER_PATH__, sep='\n')

from knn import *
from Tuning.datatune import *
from Tuning.logger import info

@jit
def data_loader(dateset_dir):
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
    for index in range(test_num):
        testcase = test_data[index]
        label = test_labels[index]
        classify_result = classfier.classify(testcase)
        info("分类返回结果为%d\t真实结果为%d" % (classify_result, label))
        if (classify_result != label):
            errorCount += 1.0
    info("总共错了%d个数据, 错误率为%f%%" % (errorCount, errorCount/test_num))

if __name__ == '__main__':
    classify_test(knn)
    classify_test(knn_sklearn)