# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'KNN/2_person_knn/'
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

#%% Function description: 打开并解析文件，对数据进行分类：1:didntLike, 2:smallDoses, 3:largeDoses
def data_loader(filename, interval, encode):
    file_array = file2array2(filename, np.str, interval, encode)
    data_array = file_array[:, 0:-1].astype(np.float)
    label_array = file_array[:, -1].astype(np.str)

    labels_dict = {'didntLike': 1, 'smallDoses':2, 'largeDoses':3}
    labels_list = [labels_dict[label] for label in label_array]

    return data_array, labels_list

"""
Function description: 使用数据集进行分类器验证
"""
@jit
def predict_validation(data_array, label_array, ratio, k):
    # 返回归一化后的矩阵,数据范围,数据最小值
    data_norm, _, _ = normalization(data_array)

    # 获得data_norm的行数, 百分之十的测试数据的个数
    val_ratio_num = int(data_array.shape[0] * ratio)

    # 分类错误计数
    errorCount = 0.0
    labels_dict = {1:'didntLike', 2:'smallDoses', 3:'largeDoses'}

    classifier = knn(k).fit(data_norm, label_array)
    for i in range(val_ratio_num):
        predict_Result = classifier.predict(data_norm[i])
        info("correction: %s, { prediction: %s, reality: %s }" % (predict_Result == label_array[i],
                                                                  labels_dict[predict_Result],
                                                                  labels_dict[label_array[i]]))
        if predict_Result != label_array[i]:
            errorCount += 1.0
    info("错误率:%f%%" %(errorCount / float(val_ratio_num) * 100))

"""
Function description:使用数据集进行分类器测试
"""
def predict_test(data_array, label_array, k):
    # 返回归一化后的矩阵,数据范围,数据最小值
    data_norm, ranges, minVals = normalization(data_array)

    # 三维特征用户输入
    precentTats = float(input("hobby1 times:"))
    ffMiles = float(input("hobby2 times:"))
    iceCream = float(input("hobby3 times:"))

    # 测试集归一化
    inArr = np.array([ffMiles, precentTats, iceCream])
    norminArr = (inArr - minVals) / ranges

    # 返回分类结果
    classifier = knn(k).fit(data_norm, label_array)
    predict_Result = classifier.predict(norminArr)

    resultList = ['didntLike','smallDoses','largeDoses']
    info("prediction: %s" % (resultList[predict_Result-1]))

def showdatas(data_array, label_array):
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    canvas, figure = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    Legends = [['didntLike', 'smallDoses', 'largeDoses'], ['black', 'orange', 'red']]
    LabelsColors = [Legends[1][i-1] for i in label_array]

    # 以data_array矩阵的第一列(hobby1)、第二列(hobby2)数据画散点图
    plt_draw2d(figure[0][0], data_array[:,0], data_array[:,1], LabelsColors,
              'scatter', Legends, u'', u'hobby1 times', u'hobby2 times')

    # 以data_array矩阵的第二列(hobby2)、第三列(hobby3)数据画散点图
    plt_draw2d(figure[0][1], data_array[:,1], data_array[:,2], LabelsColors,
              'scatter', Legends, u'', u'hobby2 times', u'hobby3 times')

    # 以data_array矩阵的第一列(hobby1)、第三列(hobby3)数据画散点图
    plt_draw2d(figure[1][0], data_array[:,0], data_array[:,2], LabelsColors,
               'scatter', Legends, u'', u'hobby1 times', u'hobby3 times')

    plt.show(canvas)

#%%
if __name__ == '__main__':
    datasetcsv = __F_PATH__ + "datingTestSet.csv"
    data_array, label_array = data_loader(datasetcsv, '\t', 'utf-8')
    showdatas(data_array, label_array)
    predict_validation(data_array, label_array, 0.1, 20)
    predict_test(data_array, label_array, 20)