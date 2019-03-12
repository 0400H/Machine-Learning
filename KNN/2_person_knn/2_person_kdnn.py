# -*- coding: UTF-8 -*-

import os
from sys import path
__Father_Root__ = os.path.dirname(__file__) + '/'
__Project_Root__ = os.path.dirname(__Father_Root__ + '../../')
path.append(__Project_Root__)

from DataTune.datatune import *
import numpy as np
import operator

"""
Function description: 打开并解析文件，对数据进行分类：1:didntLike, 2:smallDoses, 3:largeDoses
"""
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
def classify_love(inX, dataSet, labels, K):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX， 共1次(横向), 行向量方向上重复inX， 共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #平方后元素相加,axis==0列相加,axis==1行相加
    sqDistances = np.square(diffMat).sum(axis=1)
    #开方,计算出距离
    distances = np.power(sqDistances, 0.5)
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(K):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #计算类别次数， dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #reverse==True降序排序字典, itemgetter(0) 与 itemgetter(1)分别根据字典的键和值进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print('label,times:', sortedClassCount)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

"""
Function description: 使用数据集进行分类器验证
"""
def classify_verification(filename, ratio, K):
    data_array, label_array = data_loader(filename, '\t', 'utf-8')
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(data_array)

    #获得normMat的行数, 百分之十的测试数据的个数
    ver_num = int(data_array.shape[0] * ratio)

    #分类错误计数
    errorCount = 0.0
    labels_dict = {1:'didntLike', 2:'smallDoses', 3:'largeDoses'}
    for i in range(ver_num):
        #前 ver_num 个数据作为测试集
        classify_Result = classify_love(normMat[i], normMat, label_array, K)
        print("correction: %s, prediction: %s, reality: %s" % (classify_Result == label_array[i],
                                                               labels_dict[classify_Result],
                                                               labels_dict[label_array[i]]))
        if classify_Result != label_array[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount / float(ver_num) * 100))

"""
Function description:使用数据集进行分类器测试
"""
def classify_test(filename, K):
    #打开并处理数据
    data_array, label_array = data_loader(filename, '\t', 'utf-8')
    #训练集归一化
    normMat, ranges, minVals = autoNorm(data_array)

    #定义输出结果
    resultList = ['didntLike','smallDoses','largeDoses']

    #三维特征用户输入
    precentTats = float(input("hobby1 times:"))
    ffMiles = float(input("hobby2 times:"))
    iceCream = float(input("hobby3 times:"))
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classify_Result = classify_love(norminArr, normMat, label_array, K)
    #打印结果
    print("prediction: %s" % (resultList[classify_Result-1]))


def showdatas(data_array, label_array) :
    # fontfile = r"c:\windows\fonts\simsun.ttc"
    # fontfile = r"/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
    fontfile = r"/usr/share/fonts/opentype/dejavu-sans-mono/DejaVuSansMono.ttf"

    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    canvas, figure = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13, 8))

    LabelsColorsDict = {1:'black', 2:'orange', 3:'red'}
    LabelsColors = [LabelsColorsDict[i] for i in label_array]

    #画出散点图,以data_array矩阵的第一(hobby2)、第二列(hobby1)数据画散点数据,散点大小为15,透明度为0.5
    data2plt(figure[0][0], '00', data_array[:,0], data_array[:,1],
             fontfile, True, LabelsColors, 15, 0.5,
             u'hobby2 with hobby1', 9, 'bold', 'red',
             u'hobby2 times', 7, 'bold', 'black', 
             u'hobby1 times', 7, 'bold', 'black')

    #画出散点图,以data_array矩阵的第一(hobby2)、第三列(hobby3)数据画散点数据,散点大小为15,透明度为0.5
    data2plt(figure[0][1], '01', data_array[:,0], data_array[:,2],
             fontfile, True, LabelsColors, 15, 0.5,
             u'hobby2 with hobby3', 9, 'bold', 'red',
             u'hobby2 times', 7, 'bold', 'black', 
             u'hobby3 times', 7, 'bold', 'black')

    #画出散点图,以data_array矩阵的第二(hobby1)、第三列(hobby3)数据画散点数据,散点大小为15,透明度为0.5
    data2plt(figure[1][0], '10', data_array[:,1], data_array[:,2],
             fontfile, True, LabelsColors, 15, 0.5,
             u'hobby1 with hobby3', 9, 'bold', 'red',
             u'hobby1 times', 7, 'bold', 'black', 
             u'hobby3 times', 7, 'bold', 'black')

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
    datasetcsv = __Father_Root__ + "datingTestSet.csv"
    data_array, label_array = data_loader(datasetcsv, '\t', 'utf-8')
    showdatas(data_array, label_array)
    classify_verification(datasetcsv, 0.1, 20)
    classify_test(datasetcsv, 20)