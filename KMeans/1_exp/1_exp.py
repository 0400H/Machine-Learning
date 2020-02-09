# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'KMeans/1_exp/'
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

from kmeans import *
from Tuning.datatune import *
from Tuning.logger import info

# %% Function description: read data_array, labels from csv
def data_loader(filename, interval, encode):
    file_array = file2array2(filename, np.str, interval, encode)
    data_array = file_array[:, 0:-1].astype(np.float)
    return data_array

#%%
if __name__ == '__main__':
    datasetcsv = __F_PATH__ + "kmeans.csv"
    val_data = data_loader(datasetcsv, '\t', 'utf-8')
    test_dataset = np.array([[70920, 11.326976, 0.953952],
                             [36429, 4.15346, 1.673904],
                             [5477, 4.441871, 0.805124]])

    classifier_1 = kmeans(3, 1e-1).fit(val_data)
    classifier_2 = kmeans_sklearn(3, 1e-1).fit(val_data)

    for testcase in test_dataset:
        category_1 = classifier_1.predict(testcase)
        category_2 = classifier_2.predict(testcase)
        info(testcase, ':', category_1[0], category_2[0])