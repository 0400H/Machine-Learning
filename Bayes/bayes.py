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
from Tuning.func import sigmoid
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
函数说明: 朴素贝叶斯分类器训练函数
Parameters:
    data_mask_list - 训练文档矩阵，即Entry2GlossaryMask返回的glossary_mask构成的矩阵
    class_vector - 训练类别标签向量，即DataLoader返回的class_vector
Returns:
    p_word_0 - entry类别为1时, glossary中的每个单词类别为0的概率
    p_word_1 - entry类别为0时，glossary中的每个单词类别为1的概率
    p_1_entry - entry_list中词条类别为1的概率
"""
class native_bayes(object):
    @jit
    def __init__(self, data_mask_list, class_vector):
        entry_num = len(data_mask_list)
        self.p_1_entry = sum(class_vector)/float(entry_num)
        class_0_mask_sum, class_1_mask_sum = 0.0, 0.0
        class_0_mask_list_sum, class_1_mask_list_sum = 0.0, 0.0
        for i in range(entry_num):
            if class_vector[i] == 1:
                class_1_mask_list_sum += data_mask_list[i]
                class_1_mask_sum += sum(data_mask_list[i])
            else:
                class_0_mask_list_sum += data_mask_list[i]
                class_0_mask_sum += sum(data_mask_list[i])
        # P(w0|1) + P(w1|1) + P(w2|1) ···
        # P(w0|0) + P(w1|0) + P(w2|0) ···
        self.p_word_0 = class_0_mask_list_sum / class_0_mask_sum
        self.p_word_1 = class_1_mask_list_sum / class_1_mask_sum
        return None

    """
        to deal with the 0 probability problem
        useing sigmoid to quantized input(>=0) to [0.5, 1.0]
        or using Laplace smoothing
    """
    def classify(self, entry):
        p1 = reduce(lambda x, y: x * y, sigmoid(entry * self.p_word_1)) * self.p_1_entry
        p0 = reduce(lambda x, y: x * y, sigmoid(entry * self.p_word_0)) * (1.0 - self.p_1_entry)
        # info('p0:', p0， 'p1:', p1)
        return (0, 1)[p1 > p0]

pass

"""
函数说明: 验证朴素贝叶斯分类器
"""
@jit
def NB_val():
    entry_list, class_vector = DataLoader()
    myGlossary = EntryList2Glossary(entry_list)
    entry_num = len(entry_list)

    entry_mask_list = []
    for entry in entry_list:
        entry_mask_list.append(Entry2GlossaryMask(myGlossary, entry))
    classifier= native_bayes(np.array(entry_mask_list), np.array(class_vector))

    count = 0
    for index in range(entry_num):
        data_array = np.array(Entry2GlossaryMask(myGlossary, entry_list[index]))
        classify_Result = classifier.classify(data_array)
        reality = class_vector[index]
        if classify_Result == class_vector[index]:
            count += 1
        info('validation, real: %d, predict: %d, accuracy: %f' % (reality, classify_Result, count/(index + 1)))

"""
函数说明: 测试朴素贝叶斯分类器
"""
@jit
def NB_test(test_entry_list):
    entry_list, class_vector = DataLoader()
    myGlossary = EntryList2Glossary(entry_list)

    entry_mask_list = []
    for entry in entry_list:
        entry_mask_list.append(Entry2GlossaryMask(myGlossary, entry))
    classifier= native_bayes(np.array(entry_mask_list), np.array(class_vector))

    for testcase in test_entry_list:
        testarray = np.array(Entry2GlossaryMask(myGlossary, testcase))
        classify_Result = classifier.classify(testarray)
        if classify_Result:
            info(testcase, ' 属于类 1')
        else:
            info(testcase, ' 属于类 0')

#%%
if __name__ == '__main__':
    test_entry_list = [
        ['love', 'my', 'dalmation'],
        ['stupid', 'garbage'],
        ['love', 'my', 'dalmation', 'stupid', 'garbage'],
        ['love', 'my', 'dalmation', 'stupid'],
    ]
    NB_val()
    NB_test(test_entry_list)