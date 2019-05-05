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
        from Tuning.datatune import *
    except ModuleNotFoundError:
        __ML_PATH__ = os.getcwd() + '/'
        __FATHER_PATH__ = __ML_PATH__ + 'Bayes/1_easy_native_bayes/'
        pass
    pass
__ALGO_PATH__ = os.path.abspath(__ML_PATH__ + '/Bayes')
sys.path.append(__ML_PATH__)
sys.path.append(__ALGO_PATH__)
print(__ML_PATH__, __ALGO_PATH__, __FATHER_PATH__, sep='\n')

from bayes import native_bayes
from Tuning.datatune import *
from Tuning.logger import info

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
函数说明: 验证朴素贝叶斯分类器
"""
@jit
def native_bayes_val(with_log):
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
        classify_Result = classifier.classify(data_array, with_log)
        reality = class_vector[index]
        if classify_Result == class_vector[index]:
            count += 1
        info('validation { real: %d, predict: %d, accuracy: %f }' % (reality, classify_Result, count/(index + 1)))

"""
函数说明: 测试朴素贝叶斯分类器
"""
@jit
def native_bayes_test(test_entry_list, with_log):
    entry_list, class_vector = DataLoader()
    myGlossary = EntryList2Glossary(entry_list)

    entry_mask_list = []
    for entry in entry_list:
        entry_mask_list.append(Entry2GlossaryMask(myGlossary, entry))
    classifier= native_bayes(np.array(entry_mask_list), np.array(class_vector))

    for testcase in test_entry_list:
        testarray = np.array(Entry2GlossaryMask(myGlossary, testcase))
        classify_Result = classifier.classify(testarray, with_log)
        if classify_Result:
            info('test: ', testcase, ' 属于类 1')
        else:
            info('test: ', testcase, ' 属于类 0')

#%%
if __name__ == '__main__':
    test_entry_list = [
        ['love', 'my', 'dalmation'],
        ['stupid', 'garbage', 'dog'],
        ['love', 'my', 'dalmation', 'stupid'],
        ['love', 'my', 'dalmation', 'stupid', 'garbage', 'dog'],
    ]
    native_bayes_val(True)
    native_bayes_test(test_entry_list, True)
    native_bayes_val(False)
    native_bayes_test(test_entry_list, False)