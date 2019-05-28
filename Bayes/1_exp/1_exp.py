# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys
try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Bayes/1_exp/'
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

from bayes import *
from Tuning.datatune import *
from Tuning.logger import info
import random

#%%
def DataLoader():
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
    train_category_vector = [0, 1, 0, 1, 0, 1]
    return train_entry_list, train_category_vector, test_entry_list

"""
函数说明: 测试朴素贝叶斯分类器
"""
@jit
def nb_test(train_entry_list, train_category_vector, test_entry_list, with_log, model='bow'):
    glossary = list(EntryList2Glossary(train_entry_list).keys())

    entry_mask_list = []
    for entry in train_entry_list:
        entry_mask_list.append(EntryEncode(glossary, entry, model))
    classifier = native_bayes()
    classifier.fit(entry_mask_list, train_category_vector)

    classifier_sklearn = native_bayes_sklearn()
    classifier_sklearn.fit(entry_mask_list, train_category_vector)

    for test_entry in test_entry_list:
        test_entry_mask = EntryEncode(glossary, test_entry, model)
        predict_loc = classifier.predict(test_entry_mask, with_log)
        predict_sklearn = classifier_sklearn.predict(test_entry_mask)
        result = predict_loc == predict_sklearn
        log_format = 'test {{ with_log: {}, model: {}, result: {}, sklearn: {}, loc: {}, entry: {} }}'
        info(log_format.format(with_log, model, result, predict_sklearn, predict_loc, test_entry))

#%%
if __name__ == '__main__':
    train_entry_list, train_category_vector, test_entry_list = DataLoader()
    nb_test(train_entry_list, train_category_vector, test_entry_list, False, 'sow')
    nb_test(train_entry_list, train_category_vector, test_entry_list, True, 'sow')
    nb_test(train_entry_list, train_category_vector, test_entry_list, False, 'bow')
    nb_test(train_entry_list, train_category_vector, test_entry_list, True, 'bow')