# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'KNN/1_exp/'
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

# %% Function description: read data_array, labels from csv
def data_loader(filename, nop, interval, batch_dim, feature_dim, label_dim):
    (h_start, h_end), (w_start, w_end) = batch_dim, feature_dim
    data_array = file2array1(filename, np.str, nop, interval)
    data = data2matrix(data_array, h_start, h_end, w_start, w_end, np.float)
    labels = data2col(data_array, h_start, h_end, label_dim, label_dim+1, np.str)
    return data, labels

#%%
if __name__ == '__main__':
    val_data, val_labels = data_loader(__F_PATH__ + 'knn.csv', '#', ',', (0, 9), (1, 3), 0)
    test_dataset = np.array([[3, 11], [31, 42], [79, 80]])

    classifier_1 = knn(3).fit(val_data, val_labels)
    classifier_2 = knn_sklearn(3).fit(val_data, val_labels)

    for testcase in test_dataset:
        category_1 = classifier_1.predict(testcase)
        category_2 = classifier_2.predict(testcase)
        info(testcase, ':', category_1, category_2[0])