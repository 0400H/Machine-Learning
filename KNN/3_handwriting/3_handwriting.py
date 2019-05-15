# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'KNN/3_handwriting/'
    sys.path.append(__ML_PATH__)
    from Tuning.datatune import *
except ModuleNotFoundError:
    __F_PATH__ = os.getcwd() + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../../')
    pass
__ALGO_PATH__ = os.path.abspath(__F_PATH__ + '../')
sys.path.append(__ML_PATH__)
sys.path.append(__ALGO_PATH__)
print(__ML_PATH__, __ALGO_PATH__, __F_PATH__, sep='\n')

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
        predict_number = int(filename.split('_')[0])
        labels.append(predict_number)
        data_array[i] = img2col(dateset_dir + '/' + filename, 0, 32, 0, 32, np.int, 'utf-8')

    return data_array, labels

# 手写数字分类测试
@jit
def predict_test(class_classifier):
    ver_data, ver_labels = data_loader(__F_PATH__ + 'trainingDigits')
    test_data, test_labels = data_loader(__F_PATH__ + 'testDigits')

    classifier = class_classifier(ver_data, ver_labels, 3)

    errorCount = 0.0
    test_num = len(test_labels)
    for index in range(test_num):
        testcase = test_data[index]
        label = test_labels[index]
        predict_result = classifier.predict(testcase)
        info("分类返回结果为%d, 真实结果为%d" % (predict_result, label))
        if (predict_result != label):
            errorCount += 1.0
    info("总共错了%d个数据, 错误率为%f%%" % (errorCount, errorCount/test_num))

if __name__ == '__main__':
    predict_test(knn)
    predict_test(knn_sklearn)