# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'KMeans/'
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
from Tuning.logger import info
from sklearn.cluster import KMeans
import numpy as np

"""
Parameters:
Returns:
"""

class kmeans_sklearn(object):
    def __init__(self, k, loss=1e-1):
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        self._kmeans_kernel = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=10000,
                                  tol=loss, precompute_distances='auto', verbose=0,
                                  random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
        return None

    def fit(self, data_array):
        self._kmeans_kernel.fit(data_array)
        return self

    def predict(self, test_data):
        return self._kmeans_kernel.predict(test_data.reshape(1, -1))

    def score(self, val_data, val_label):
        return self._kmeans_kernel.score(val_data, val_label)

class kmeans(object):
    def __init__(self, k, loss=1e-1):
        self._K = k
        self._loss = loss
        return None

    def fit(self, data_array):
        self._data = data_array
        self._num = self._data.shape[0]
        self._attributes = self._data.shape[1]
        self._random_k = np.array([self._data[idx] for idx in np.random.randint(0, self._num, self._K, dtype=np.int)])

        loss = self._loss + 1e-10
        label = np.zeros(self._num, dtype=np.int)
        while loss > self._loss:
            loss = 0
            for idx in range(self._num):
                distance = np.zeros(self._K)
                for k_idx in range(self._K):
                    distance[k_idx] = l2_distance(self._data[idx], self._random_k[k_idx])
                label[idx] = np.argmin(distance)

            for k_idx in range(self._K):
                k_data = self._data[label == k_idx]
                k_center = np.sum(k_data, axis=0) / len(k_data)
                loss += l2_distance(k_center, self._random_k[k_idx])
                self._random_k[k_idx] = k_center
        return self

    def predict(self, test_data):
        length = len(test_data)
        label = np.zeros(length, dtype=np.int)
        distance = np.zeros(shape=(length, self._K))
        for idx in range(length):
            for k_idx in range(self._K):
                distance[idx][k_idx] = l2_distance(test_data[idx], self._random_k[k_idx])
        label = np.argmin(distance, axis=1)
        return label

    def score(self, val_data, val_label):
        return None