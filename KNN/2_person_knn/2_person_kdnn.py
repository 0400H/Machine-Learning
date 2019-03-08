# -*- coding: UTF-8 -*-

from sys import path
path.append("../../")
from DataTune.datatune import *

import numpy as np
import operator

"""
Function description:
    打开并解析文件，对数据进行分类：1:didntLike,2:smallDoses,3:largeDoses
"""
def file2data_label(filename, interval, encode) :
    file_ndarray = file2ndarray(filename, interval, encode)
    data_ndarray = file_ndarray[:, 0:-1].astype(np.float)
    label_ndarray = file_ndarray[:, -1].astype(np.str)

    labels_dict = {'didntLike': 1, 'smallDoses':2, 'largeDoses':3}
    labels_list = []

    for label in label_ndarray:
        labels_list.append(labels_dict[label])

    return data_ndarray, labels_list

"""
Function description:
    kNN算法, 分类器
Parameters:
    inX     - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes   - 分类标签
    k       - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classify_love(inX, dataSet, labels, K):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加,axis==0列相加,axis==1行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方,计算出距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(K):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #计算类别次数， dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #reverse==True降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print('label,times:', sortedClassCount)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

"""
Function description:使用数据集进行分类器验证
"""
def classify_verification(filename, ratio, K):
    data_ndarray, label_ndarray = file2data_label(filename, '\t', 'utf-8')
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(data_ndarray)
    #获得normMat的行数
    num_of_row = normMat.shape[0]
    #百分之十的测试数据的个数
    num_of_testcase = int(num_of_row * ratio)
    #分类错误计数
    errorCount = 0.0

    labels_dict = {1:'didntLike', 2:'smallDoses', 3:'largeDoses'}
    for i in range(num_of_testcase):
        #前num_of_testcase个数据作为测试集,后num_of_testcase个数据作为训练集
        classifierResult = classify_love(normMat[i,:], normMat[num_of_testcase:num_of_row,:], 
                                        label_ndarray[num_of_testcase:num_of_row], K)
        print("correction: %s, prediction: %s, reality: %s" % (classifierResult == label_ndarray[i],
                                                               labels_dict[classifierResult],
                                                               labels_dict[label_ndarray[i]]))
        if classifierResult != label_ndarray[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(num_of_testcase)*100))

"""
Function description:使用数据集进行分类器测试
"""
def classify_test(filename, K):
    #打开并处理数据
    data_ndarray, label_ndarray = file2data_label(filename, '\t', 'utf-8')
    #训练集归一化
    normMat, ranges, minVals = autoNorm(data_ndarray)

    #定义输出结果
    resultList = ['didntLike','smallDoses','largeDoses']

    #三维特征用户输入
    precentTats = float(input("hobby1 ratio a year:"))
    ffMiles = float(input("hobby2 ratio a year:"))
    iceCream = float(input("hobby3 ratio a year:"))
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify_love(norminArr, normMat, label_ndarray, K)
    #打印结果
    print("prediction: %s" % (resultList[classifierResult-1]))


def showdatas(data_ndarray, label_ndarray) :
    # fontfile = r"c:\windows\fonts\simsun.ttc"
    fontfile = r"/usr/share/fonts/dejavu/DejaVuSansMono.ttf"

    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    canvas, figure = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    LabelsColorsDict = {1:'black', 2:'orange', 3:'red'}
    LabelsColors = [LabelsColorsDict[i] for i in label_ndarray]

    #画出散点图,以data_ndarray矩阵的第一(hobby2)、第二列(hobby1)数据画散点数据,散点大小为15,透明度为0.5
    data2plt(figure[0][0], '00', data_ndarray[:,0], data_ndarray[:,1],
             fontfile, True, LabelsColors, 15, 0.5,
             u'hobby2 / hobby1 ratio', 9, 'bold', 'red',
             u'hobby2 time ratio a year', 7, 'bold', 'black', 
             u'hobby1 time ratio a year', 7, 'bold', 'black')

    #画出散点图,以data_ndarray矩阵的第一(hobby2)、第三列(hobby3)数据画散点数据,散点大小为15,透明度为0.5
    data2plt(figure[0][1], '01', data_ndarray[:,0], data_ndarray[:,2],
             fontfile, True, LabelsColors, 15, 0.5,
             u'hobby2 / hobby3 ratio', 9, 'bold', 'red',
             u'hobby2 time ratio a year', 7, 'bold', 'black', 
             u'hobby3 time ratio a year', 7, 'bold', 'black')

    #画出散点图,以data_ndarray矩阵的第二(hobby1)、第三列(hobby3)数据画散点数据,散点大小为15,透明度为0.5
    data2plt(figure[1][0], '10', data_ndarray[:,1], data_ndarray[:,2],
             fontfile, True, LabelsColors, 15, 0.5,
             u'hobby1 / hobby3 ratio', 9, 'bold', 'red',
             u'hobby1 time ratio a year', 7, 'bold', 'black', 
             u'hobby3 time ratio a year', 7, 'bold', 'black')

    #设置图例
    didntLike = get_marker_Line2D('black', 6, 'didntLike')
    smallDoses = get_marker_Line2D('orange', 6, 'smallDoses')
    largeDoses = get_marker_Line2D('red', 6, 'largeDoses')
    #添加图例
    handle = [didntLike, smallDoses, largeDoses]
    add_legend(figure[0][0], handle)
    add_legend(figure[0][1], handle)
    add_legend(figure[1][0], handle)

    #显示图片
    show_pyplot(plt)

if __name__ == '__main__':
    datasetcsv = "datingTestSet.csv"
    data_ndarray, label_ndarray = file2data_label(datasetcsv, '\t', 'utf-8')
    showdatas(data_ndarray, label_ndarray)
    classify_verification(datasetcsv, 0.1, 20)
    classify_test(datasetcsv, 20)