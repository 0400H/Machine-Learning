# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __F_PATH__ = os.path.dirname(os.path.abspath(__file__)) + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../../')
except NameError:
    try:
        __F_PATH__ = os.getcwd() + '/'
        __ML_PATH__ = os.path.abspath(__F_PATH__ + '../../')
        from Tuning.datatune import *
    except ModuleNotFoundError:
        __ML_PATH__ = os.getcwd() + '/'
        __F_PATH__ = __ML_PATH__ + 'Bayes/1_set_of_words/'
        pass
    pass
__ALGO_PATH__ = os.path.abspath(__ML_PATH__ + '/Bayes')
sys.path.append(__ML_PATH__)
sys.path.append(__ALGO_PATH__)
print(__ML_PATH__, __ALGO_PATH__, __F_PATH__, sep='\n')

from bayes import native_bayes
from Tuning.datatune import *
from Tuning.logger import info
import random
import re

#%%
def DataLoader1():
    test_entry_list = [['love', 'my', 'dalmation'],
                       ['stupid', 'garbage', 'dog'],
                       ['love', 'my', 'dalmation', 'stupid'],
                       ['love', 'my', 'dalmation', 'stupid', 'garbage', 'dog']]
    train_entry_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    train_category_vector = [0, 1, 0, 1, 0, 1] # 二分类
    return train_entry_list, train_category_vector, test_entry_list

def DataLoader2():
    entry_list, category_vector = [], []
    for i in range(1, 26):
        text = open(__ALGO_PATH__ + './email/spam/%d.txt' % i, 'r').read()
        wordList = TextParse(text)
        entry_list.append(wordList)
        category_vector.append(1)          # 标记垃圾邮件，1表示垃圾文件
        text = open(__ALGO_PATH__ + './email/ham/%d.txt' % i, 'r').read()
        wordList = TextParse(text)
        entry_list.append(wordList)
        category_vector.append(0)

    return entry_list, category_vector

def EntryList2Glossary(entry_list):
    glossary = set([])
    for entry in entry_list:
        glossary = glossary | set(entry)
    return list(glossary)

# 使用词汇表的mask(0 or 1)表示词条
def Entry2GlossaryMask(glossary, entry):
    glossary_mask = [0 for i in range(len(glossary))]
    for word in entry:
        if word in glossary:
            glossary_mask[glossary.index(word)] = 1
        else:
            info("the word: %s is not in my glossary!" % word)
    return glossary_mask

# 接收一个大字符串并将其解析为字符串列表
def TextParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    entry = [tok.lower() for tok in listOfTokens if len(tok) > 2]           #除了单个字母，例如大写的I，其它单词变成小写
    return entry

"""
函数说明: 验证朴素贝叶斯分类器
"""
@jit
def sow_val(entry_list, category_vector, with_log, rate=0.2):
    myGlossary = EntryList2Glossary(entry_list)
    entry_num = len(entry_list)
    val_num = int(rate*entry_num)

    TestSetIndex = []
    TrainSetIndex = list(range(entry_num))
    val_count = 0
    while val_count < val_num:
        randIndex = int(random.uniform(0, entry_num))
        if randIndex in TestSetIndex:
            continue
        else:
            val_count += 1
            TestSetIndex.append(randIndex)
            TrainSetIndex.remove(randIndex)

    train_entry_mask_list, train_category_vector = [], []
    for index in TrainSetIndex:
        train_entry_mask_list.append(Entry2GlossaryMask(myGlossary, entry_list[index]))
        train_category_vector.append(category_vector[index])

    classifier= native_bayes(np.array(train_entry_mask_list), np.array(train_category_vector))

    right_count, length_count = 0, 0
    for index in TestSetIndex:
        val_entry_mask = Entry2GlossaryMask(myGlossary, entry_list[index])
        val_data_array = np.array(val_entry_mask)
        val_reality = category_vector[index]

        classify_result = classifier.classify(val_data_array, with_log)

        log_format = 'validation { real: %d, predict: %d, accuracy: %f'
        length_count += 1
        if classify_result == val_reality:
            right_count += 1.0
            info(log_format % (val_reality, classify_result, right_count/length_count), 'with_log:', with_log, '}')
        else:
            info(log_format % (val_reality, classify_result, right_count/length_count), 'with_log:', with_log, '}')
            info(entry_list[index])

"""
函数说明: 测试朴素贝叶斯分类器
"""
@jit
def sow_test(train_entry_list, train_category_vector, test_entry_list, with_log):
    myGlossary = EntryList2Glossary(train_entry_list)

    entry_mask_list = []
    for entry in train_entry_list:
        entry_mask_list.append(Entry2GlossaryMask(myGlossary, entry))
    classifier= native_bayes(np.array(entry_mask_list), np.array(train_category_vector))

    for testcase in test_entry_list:
        testarray = np.array(Entry2GlossaryMask(myGlossary, testcase))
        classify_result = classifier.classify(testarray, with_log)
        if classify_result:
            info('test:', '属于类 1', 'with_log:', with_log, testcase)
        else:
            info('test:', '属于类 0', 'with_log:', with_log, testcase)

#%%
if __name__ == '__main__':
    train_entry_list, train_category_vector, test_entry_list = DataLoader1()
    sow_test(train_entry_list, train_category_vector, test_entry_list, False)
    sow_test(train_entry_list, train_category_vector, test_entry_list, True)

    train_entry_list, train_category_vector = DataLoader2()
    sow_val(train_entry_list, train_category_vector, False)
    sow_val(train_entry_list, train_category_vector, True)