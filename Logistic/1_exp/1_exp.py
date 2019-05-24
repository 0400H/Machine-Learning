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
    label_array = file_array[:, -1].reshape(-1).astype(np.float)
    data_array = np.ones(shape=(len(file_array), 3), dtype=np.float)
    # X_i {x1, x2, 1} W_i {w1, w2, w3(bias)}
    data_array[:, :-1] = file_array[:, :-1].astype(np.float)
    return data_array, label_array

"""
函数说明:绘制数据集
"""
def plotBestFit(data_matrix, label_matrix, weights):
    x1cord1, x2cord1 = [], []                                               #正样本
    x1cord2, x2cord2 = [], []                                               #负样本
    for index in range(len(label_matrix)):                                  #根据数据集标签进行分类
        if int(label_matrix[index]) == 1:
            x1cord1.append(data_matrix[index, 0])
            x2cord1.append(data_matrix[index, 1])
        else:
            x1cord2.append(data_matrix[index, 0])
            x2cord2.append(data_matrix[index, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1cord1, x2cord1, s = 20, c = 'red', marker = 's', alpha=.5)
    ax.scatter(x1cord2, x2cord2, s = 20, c = 'green', marker = 's', alpha=.5)
    x1 = np.arange(-3.0, 3.0, 0.1)
    x2 = -(weights[2] + weights[0] * x1) / weights[1]
    ax.plot(x1, x2)
    plt.title('BestFit')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

# def plotBestFit(data_matrix, label_matrix, weights):
#     # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
#     canvas, figure = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

#     Legends = [['didntLike', 'smallDoses', 'largeDoses'], ['black', 'orange', 'red']]
#     LabelsColors = [Legends[1][i-1] for i in label_array]

#     # 以data_array矩阵的第一列(hobby1)、第二列(hobby2)数据画散点图
#     plt_draw2d(figure[0][0], data_array[:,0], data_array[:,1], LabelsColors,
#               'scatter', Legends, u'', u'hobby1 times', u'hobby2 times')
#     plt.show()

#%%
if __name__ == '__main__':
    data_matrix, label_matrix = data_loader(__F_PATH__ + 'testSet.csv')

    classifier = logistic()
    weight, num_iter = classifier.fit(data_matrix, label_matrix, 1e-2, 1e-2)
    predict_result = classifier.predict(data_matrix)
    error_rate = np.average(predict_result - label_matrix.reshape(-1))
    info('error_rate: {}, iteration: {}'.format(error_rate, num_iter))

    plotBestFit(data_matrix, label_matrix, weight)