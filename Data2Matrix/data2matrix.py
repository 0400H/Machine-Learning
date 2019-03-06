# -*- coding: UTF-8 -*-

import numpy as np

'''
    Function description:
        read dataset fpom csv
    Parameters:
        filename
        label_dim
        batch_dim
    Returns:
        data
        label
'''
def NumpyReadData(filename, h_start, h_end, w_start, w_end, in_dtype = np.str, out_dtype = np.str):
    dataset = np.loadtxt(filename, dtype = in_dtype, comments = '#', delimiter = ',')
    select_data = dataset[h_start:h_end, w_start:w_end]

    if w_end <= w_start + 1 :
        select_data = dataset[h_start:h_end, w_start]

    data = select_data.astype(out_dtype)
    return data

def ReadDataSet2Ndarray(filename, batch_dim, feature_dim, label_dim):
    (h_start, h_end), (w_start, w_end) = batch_dim, feature_dim
    features = NumpyReadData(filename, h_start, h_end, w_start, w_end, np.str, np.float)
    labels = NumpyReadData(filename, h_start, h_end, label_dim, label_dim)
    return features, labels

"""
Function description:
    打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
"""
def file2matrix(filename, encode):
    #打开文件,此次应指定编码，
    fp = open(filename, 'r', encoding = encode)
    #按行读取文件所有内容到list
    File2List = fp.readlines()
    #针对有BOM的UTF-8文本，应该去掉BOM，否则后面会引发错误。
    File2List[0] = File2List[0].lstrip('\ufeff')
    #得到文件行数
    NumOfLines = len(File2List)
    #返回的NumPy矩阵, 解析完成的数据:number of lines行, 3列
    returnMat = np.zeros((NumOfLines, 3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in File2List:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listfpomLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listfpomLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listfpomLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listfpomLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listfpomLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector