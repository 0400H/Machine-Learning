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
from Tuning.logger import info
from functools import reduce

#%%
def DataLoader():
    entry_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1] # 二分类
    return entry_list, class_vector

def EntryList2Glossary(entry_list):
    word_set = set([])
    for entry in entry_list:
        word_set = word_set | set(entry)
    return list(word_set)


# 使用词汇表的mask(0 or 1)表示词条
def Entry2GlossaryMask(glossary, entry):
    glossary_mask = [0 for i in range(len(glossary))]
    for word in entry:
        if word in glossary:
            glossary_mask[glossary.index(word)] = 1
        else: info("the word: %s is not in my Wordulary!" % word)
    return glossary_mask


"""
函数说明:朴素贝叶斯分类器训练函数
Parameters:
    data_mask_list - 训练文档矩阵，即Entry2GlossaryMask返回的glossary_mask构成的矩阵
    class_vector - 训练类别标签向量，即DataLoader返回的class_vector
Returns:
    p_word_0 - entry类别为1时, glossary中的每个单词类别为0的概率
    p_word_1 - entry类别为0时，glossary中的每个单词类别为1的概率
    p_1_entry - entry_list中词条类别为1的概率
"""
@jit
def trainNB0(data_mask_list, class_vector):
    entry_num = len(data_mask_list)                            # 文档词条的数目
    p_1_entry = sum(class_vector)/float(entry_num)             # 文档中词条类别为1的概率
    class_0_mask_sum, class_1_mask_sum = 0.0, 0.0
    class_0_mask_list_sum, class_1_mask_list_sum = 0.0, 0.0
    for i in range(entry_num):
        if class_vector[i] == 1:
            class_1_mask_list_sum += data_mask_list[i]
            class_1_mask_sum += sum(data_mask_list[i])
        else:
            class_0_mask_list_sum += data_mask_list[i]
            class_0_mask_sum += sum(data_mask_list[i])
    # P(w0|1) + P(w1|1) + P(w2|1) ···; P(w0|0) + P(w1|0) + P(w2|0) ···
    p_word_0 = class_0_mask_list_sum / class_0_mask_sum
    p_word_1 = class_1_mask_list_sum / class_1_mask_sum
    return p_word_0, p_word_1, p_1_entry

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
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # info(p0Vec)
    # info(p1Vec)
    p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1
    p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1.0 - pClass1)
    info('p0:', p0)
    info('p1:', p1)
    if p1 > p0:
        return 1
    else: 
        return 0

"""
函数说明:测试朴素贝叶斯分类器
"""
# @jit
def testingNB():
    entry_list, class_vector = DataLoader()
    myGlossary = EntryList2Glossary(entry_list)

    # test_entry_list = [
    #     ['love', 'my', 'dalmation'],
    #     ['stupid', 'garbage'],
    # ]

    test_entry_list=[
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    entry_mask_list = []
    for entry in entry_list:
        entry_mask_list.append(Entry2GlossaryMask(myGlossary, entry))

    p_word_0, p_word_1, p_1_entry = trainNB0(np.array(entry_mask_list), np.array(class_vector))

    for testcase in test_entry_list:
        testarray = np.array(Entry2GlossaryMask(myGlossary, testcase))
        category = classifyNB(testarray, p_word_0, p_word_1, p_1_entry)
        if category:
            info(testcase, ' 属于类 1')
        else:
            info(testcase, ' 属于类 0')

#%%
if __name__ == '__main__':
    testingNB()