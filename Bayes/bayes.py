# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __FATHER_PATH__ = os.path.dirname(os.path.abspath(__file__)) + '/'
    __ML_PATH__ = os.path.abspath(__FATHER_PATH__ + '../')
except NameError:
    try:
        __FATHER_PATH__ = os.getcwd() + '/'
        __ML_PATH__ = os.path.abspath(__FATHER_PATH__ + '../')
        from Tuning.datatune import *
    except ModuleNotFoundError:
        __ML_PATH__ = os.getcwd() + '/'
        __FATHER_PATH__ = __ML_PATH__ + 'Bayes/'
        pass
    pass
sys.path.append(__ML_PATH__)
print(__ML_PATH__, __FATHER_PATH__, sep='\n')

from Tuning.datatune import *
from functools import reduce

#%%
# 二分类
def DataLoader():
    entry_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]
    return entry_list, class_vector

def EntryList2WordList(entry_list):
    word_set = set([])
    for entry in entry_list:
        word_set = word_set | set(entry)
    return list(word_set)


# 函数说明: 使用词汇表的mask(0 or 1)表示词条
def Entry2WordListMask(word_list, entry):
    word_list_mask = [0 for i in range(len(word_list))]
    for word in entry:
        if word in word_list:
            word_list_mask[word_list.index(word)] = 1
        else: print("the word: %s is not in my Wordulary!" % word)
    return word_list_mask


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
    trainMatrix - 训练文档矩阵，即Entry2WordListMask返回的word_list_mask构成的矩阵
    trainCategory - 训练类别标签向量，即DataLoader返回的classVec
Returns:
    p0Vect - 非的条件概率数组
    p1Vect - 侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
"""
@jit
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)                            #计算训练的文档数目
    numWords = len(trainMatrix[0])                            #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)        #文档属于侮辱类的概率
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)    #创建numpy.zeros数组,
    p0Denom = 0.0; p1Denom = 0.0                            #分母初始化为0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                            #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                                #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom                                    #相除        
    p0Vect = p0Num/p0Denom          
    return p0Vect,p1Vect,pAbusive                            #返回属于侮辱类条件的单词概率数组，属于非侮辱类条件的单词概率数组，文档属于侮辱类的概率

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 非侮辱类的条件概率数组
    p1Vec -侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1                #对应元素相乘
    p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else: 
        return 0

"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-12
"""
# @jit
def testingNB():
    listOPosts,listClasses = DataLoader()                                    #创建实验样本
    myWordList = EntryList2WordList(listOPosts)                                #创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(Entry2WordListMask(myWordList, postinDoc))                #将实验样本向量化
    A = np.array(trainMat)
    B = np.array(listClasses)
    p0V,p1V,pAb = trainNB0(A, B)        #训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']                                    #测试样本1
    thisDoc = np.array(Entry2WordListMask(myWordList, testEntry))                #测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')                                        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')                                        #执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']                                        #测试样本2

    thisDoc = np.array(Entry2WordListMask(myWordList, testEntry))                #测试样本向量化
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')                                        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非侮辱类')                                        #执行分类并打印分类结果

#%%
if __name__ == '__main__':
    testingNB()