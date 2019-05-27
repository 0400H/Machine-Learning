# -*- coding:UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Logistic/2_exp/'
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

from Tuning.math import *
from Tuning.datatune import *
from Tuning.logger import info
from sklearn.linear_model import LogisticRegression

def data_loader(filename):
    file_array = file2array2(filename, np.str, '\t', 'utf-8')
    label_array = file_array[:, -1].reshape(-1).astype(np.float)
    data_array = np.ones(shape=(len(file_array), len(file_array[0])), dtype=np.float)
    data_array[:, :-1] = file_array[:, :-1].astype(np.float)
    return data_array, label_array

"""
函数说明:使用Sklearn构建Logistic回归分类器
"""
def logistic_regression_sklearn():
    trainingSet, trainingLabels = data_loader(__F_PATH__ + 'horseColicTraining.csv')
    testSet, testLabels = data_loader(__F_PATH__ + 'horseColicTest.csv')
    classifier = LogisticRegression(solver = 'sag', max_iter = 5000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    info('accuracy: %f%%' % test_accurcy)

if __name__ == '__main__':
    logistic_regression_sklearn()