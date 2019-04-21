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
        __FATHER_PATH__ = __ML_PATH__ + 'KNN/2_person_knn/'
        pass
    pass
sys.path.append(__ML_PATH__)
print(__ML_PATH__, __FATHER_PATH__, sep='\n')

from DataTune.datatune import *
from DataTune.logger import info

#%% Function description: 打开并解析文件，对数据进行分类：1:didntLike, 2:smallDoses, 3:largeDoses
def data_loader(filename, interval, encode) :
    file_array = file2array2(filename, np.str, interval, encode)
    data_array = file_array[:, 0:-1].astype(np.float)
    label_array = file_array[:, -1].astype(np.str)

    labels_dict = {'didntLike': 1, 'smallDoses':2, 'largeDoses':3}
    labels_list = [labels_dict[label] for label in label_array]

    return data_array, labels_list

"""
Function description: kNN算法, 分类器
Parameters:
    inX     - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes   - 分类标签
    k       - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
@jit
def classify_love(inX, dataSet, labels, k):
    loss = inX - dataSet
    l2_distance = np.power(np.square(loss).sum(axis=1), 0.5)
    topk_index = l2_distance.argsort(kind='quicksort')[:k]
    topk_label = [labels[index] for index in topk_index]
    count_common = collect_np(topk_label)
    top1_label = get_topn(count_common, 1)
    return top1_label

"""
Function description: 使用数据集进行分类器验证
"""
@jit
def classify_validation(data_array, label_array, ratio, k):
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    data_norm, _, _ = normalization(data_array)

    #获得data_norm的行数, 百分之十的测试数据的个数
    val_ratio_num = int(data_array.shape[0] * ratio)

    #分类错误计数
    errorCount = 0.0
    labels_dict = {1:'didntLike', 2:'smallDoses', 3:'largeDoses'}
    for i in range(val_ratio_num):
        classify_Result = classify_love(data_norm[i], data_norm, label_array, k)
        info("correction: %s, { prediction: %s, reality: %s }" % (classify_Result == label_array[i],
                                                                  labels_dict[classify_Result],
                                                                  labels_dict[label_array[i]]))
        if classify_Result != label_array[i]:
            errorCount += 1.0
    info("错误率:%f%%" %(errorCount / float(val_ratio_num) * 100))

"""
Function description:使用数据集进行分类器测试
"""
@jit
def classify_test(data_array, label_array, K):
    #训练集归一化
    data_norm, ranges, minVals = normalization(data_array)

    #三维特征用户输入
    precentTats = float(input("hobby1 times:"))
    ffMiles = float(input("hobby2 times:"))
    iceCream = float(input("hobby3 times:"))
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classify_Result = classify_love(norminArr, data_norm, label_array, K)

    #打印结果
    resultList = ['didntLike','smallDoses','largeDoses']
    info("prediction: %s" % (resultList[classify_Result-1]))

@jit
def showdatas(data_array, label_array) :
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    canvas, figure = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13, 8))

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
    datasetcsv = __FATHER_PATH__ + "datingTestSet.csv"
    data_array, label_array = data_loader(datasetcsv, '\t', 'utf-8')
    showdatas(data_array, label_array)
    classify_validation(data_array, label_array, 0.1, 20)
    classify_test(data_array, label_array, 20)