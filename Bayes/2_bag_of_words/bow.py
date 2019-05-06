# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Bayes/2_bag_of_words/'
    sys.path.append(__ML_PATH__)
    from Tuning.datatune import *
except ModuleNotFoundError:
    __F_PATH__ = os.getcwd() + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../../')
    pass
__ALGO_PATH__ = os.path.abspath(__F_PATH__ + '../') + '/'
sys.path.append(__ML_PATH__)
sys.path.append(__ALGO_PATH__)
print(__ML_PATH__, __ALGO_PATH__, __F_PATH__, sep='\n')

import numpy as np
import random
import re

def EntryList2Glossary(entry_list):
    glossary = set([])
    for entry in entry_list:
        glossary = glossary | set(entry)
    return list(glossary)

def Entry2GlossaryMask(glossary, entry):
    glossary_mask = [0 for i in range(len(glossary))]
    for word in entry:
        if word in glossary:
            glossary_mask[glossary.index(word)] = 1
        else:
            info("the word: %s is not in my glossary!" % word)
    return glossary_mask
"""
函数说明:根据vocabList词汇表，构建词袋模型

Parameters:
	vocabList - EntryList2Glossary返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词袋模型
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-14
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)										#创建一个其中所含元素都为0的向量
    for word in inputSet:												#遍历每个词条
        if word in vocabList:											#如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec													#返回词袋模型
