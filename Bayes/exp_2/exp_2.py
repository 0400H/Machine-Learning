# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Bayes/exp_2/'
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

@jit
def DataLoader():
    entry_list, category_vector = [], []
    for i in range(1, 26):
        text = open(__F_PATH__ + 'email/spam/%d.txt' % i, 'r')
        wordList = TextParse(text.read())
        if len(wordList) != 0:
            entry_list.append(wordList)
            category_vector.append(1)            # 标记类别
            text.close()

        text = open(__F_PATH__ + 'email/ham/%d.txt' % i, 'r')
        wordList = TextParse(text.read())
        if len(wordList) != 0:
            entry_list.append(wordList)
            category_vector.append(0)
            text.close()

    return entry_list, category_vector

# 验证朴素贝叶斯分类器
@jit
def nb_val(entry_list, category_vector, with_log, val_rate=0.2, model='bow'):
    entry_num = len(entry_list)
    val_num = int(val_rate*entry_num)

    glossary = list(EntryList2Glossary(entry_list).keys())
    SetIndex = np.arange(entry_num)
    np.random.shuffle(SetIndex)
    TestSetIndex = SetIndex[:val_num]
    TrainSetIndex = SetIndex[val_num:]

    train_entry_feature_list, train_category_vector = [], []
    for index in TrainSetIndex:
        train_entry_feature = EntryEncode(glossary, entry_list[index], model)
        train_entry_feature_list.append(train_entry_feature)
        train_category_vector.append(category_vector[index])

    classifier = native_bayes()
    classifier.fit(train_entry_feature_list, train_category_vector)

    classifier_sklearn = native_bayes_sklearn()
    classifier_sklearn.fit(train_entry_feature_list, train_category_vector)

    right_count, length_count = 0, 0
    for index in TestSetIndex:
        val_entry_feature = EntryEncode(glossary, entry_list[index], model)
        val_label = category_vector[index]

        predict_loc = classifier.predict(val_entry_feature, with_log)
        predict_sklearn = classifier_sklearn.predict(val_entry_feature)

        length_count += 1
        right_count += float(predict_loc == val_label)
        log_format = 'nb_val {{ with_log: {}, model: {}, label: {}, sklearn: {}, loc: {}, accuracy: {} }}'
        info(log_format.format(with_log, model, val_label, predict_sklearn, predict_loc, right_count/length_count))

        if predict_loc != val_label:
            info(entry_list[index])

#%%
if __name__ == '__main__':
    train_entry_list, train_category_vector = DataLoader()
    nb_val(train_entry_list, train_category_vector, False, 0.2, 'sow')
    nb_val(train_entry_list, train_category_vector, True, 0.2, 'sow')
    nb_val(train_entry_list, train_category_vector, False, 0.2, 'bow')
    nb_val(train_entry_list, train_category_vector, True, 0.2, 'bow')