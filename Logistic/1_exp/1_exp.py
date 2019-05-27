# -*- coding:UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Logistic/1_exp/'
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
from logistic import *

#%%
# y = w0 * x1 + w1 * x2 + w2
def data_loader(filename):
    file_array = file2array2(filename, np.str, '\t', 'utf-8')
    label_array = file_array[:, -1].reshape(-1).astype(np.int)
    data_array = np.ones(shape=(len(file_array), len(file_array[0])), dtype=np.float)
    # X_i {x1, x2, 1} W_i {w1, w2, w3(bias)}
    data_array[:, :-1] = file_array[:, :-1].astype(np.float)
    return data_array, label_array

"""
函数说明:绘制数据集
"""
def plotBestFit(data_array, label_array, weight_historys):
    # draw predict line
    weight_size = len(weight_historys)
    feature_size = weight_historys[0].shape[1]
    canvas, figure = plt.subplots(nrows=feature_size+1, ncols=weight_size,
                                  sharex=False, sharey=False, figsize=(5, 5))

    for index in range(weight_size):
        weight_history = weight_historys[index]
        weight = weight_history[-1]

        Legends = [['False', 'True'], ['black', 'red']]
        LabelsColors = [Legends[1][i] for i in label_array]
        plt_draw2d(figure[0][index], data_array[:, 0], data_array[:, 1],
                   LabelsColors, 'scatter', Legends, u'', u'X1', u'X2', 10, 1)
        X1 = np.arange(-3.0, 3.0, 0.1)
        X2 = -(weight[2] + weight[0] * X1) / weight[1]
        plt_draw2d(figure[0][index], X1, X2, 'green', 'line',
                   Legends, u'', u'X1', u'X2', 10, 1)

        iteration = weight_history.shape[0]
        iter_array = np.arange(0, iteration, 1)
        for idx in range(feature_size):
            plt_draw2d(figure[idx+1][index], iter_array, weight_history[:, idx],
                       'red', 'line', '', '', 'iteration', 'W' + str(idx), 1, 1)
    plt_add_title(figure[0][0], 'logistic with batch gradient boost method')
    plt_add_title(figure[0][1], 'logistic with stochastic gradient boost method')
    plt.show(canvas)

def logistic_val(data_array, label_array):
    classifier = logistic()
    _, num_iter = classifier.fit(data_array, label_array, 1e-2, 1e-2, 'bgb', True)
    gb_weight_history = classifier.weight_history
    val_result = classifier.predict(data_array)
    error_rate = np.average(np.abs(val_result - label_array))
    info('error_rate: {}, iteration: {}'.format(error_rate, num_iter))

    _, num_iter = classifier.fit(data_array, label_array, 1e-2, 1e-2, 'sgb', True)
    sgb_weight_history = classifier.weight_history
    val_result = classifier.predict(data_array)
    error_rate = np.average(np.abs(val_result - label_array))
    info('error_rate: {}, iteration: {}'.format(error_rate, num_iter))

    return gb_weight_history, sgb_weight_history

#%%
if __name__ == '__main__':
    data_array, label_array = data_loader(__F_PATH__ + 'testSet.csv')
    plotBestFit(data_array, label_array, logistic_val(data_array, label_array))