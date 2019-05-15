# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Bayes/'
    sys.path.append(__ML_PATH__)
    from Tuning.datatune import *
except ModuleNotFoundError:
    __F_PATH__ = os.getcwd() + '/'
    __ML_PATH__ = os.path.abspath(__F_PATH__ + '../')
    pass
__ALGO_PATH__ = __F_PATH__
sys.path.append(__ML_PATH__)

from Tuning.datatune import *
from Tuning.logger import info
from functools import reduce
from sklearn.naive_bayes import MultinomialNB
import re

@jit
def EntryList2Glossary(entry_list):
    glossary_dict = {}
    for entry in entry_list:
        for word in entry:
            if word in glossary_dict.keys():
                glossary_dict[word] += 1
            else:
                glossary_dict[word] = 1
    return glossary_dict

# set of words and bag of words model
@jit
def EntryEncode(glossary, entry, model='bow'):
    glossary_mask = [0] * len(glossary)
    entry_set = set(entry)
    for index in range(len(glossary)):
        word = glossary[index]
        if word in entry_set:
            if model == 'sow':
                glossary_mask[index] = 1
            elif model == 'bow':
                glossary_mask[index] += 1
            else:
                raise RuntimeError('unknown model: %s' % model)
    return glossary_mask

# 接收一个大字符串并将其解析为字符串列表
@jit
def TextParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(pattern=r'\W*', string=bigString)               #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    entry = []
    for tok in listOfTokens:
        if len(tok) > 2:
            entry.append(tok.lower())
        else:
            continue
    return entry

"""
函数说明: 朴素贝叶斯分类器训练函数
Parameters:
    entry_feature_list - 训练文档矩阵，即EntryEncode返回的glossary_mask构成的矩阵
    category_vector - 训练类别标签向量，即DataLoader返回的category_vector
Returns:
    p_word_0 - 对类别为1的所有entry, 统计glossary中的每个单词类出现的概率
    p_word_1 - 对类别为0的所有entry, 统计glossary中的每个单词类出现的概率
    p_1_entry_feature - entry_list中词条类别为1的概率
"""
class native_bayes(object):
    """
        to deal with the 0 probability problem
        using Laplace smoothing to quantized input(>=0) to (0.0, 1.0]
    """
    @jit
    def fit(self, entry_feature_list, category_vector, laplace_smooth=True):
        entry_num = len(entry_feature_list)
        self.p_1_entry_feature = np.sum(category_vector)/entry_num

        entry_feature_list = np.array(entry_feature_list, dtype=np.float)
        category_vector = np.array(category_vector)

        laplace = (0, 1)[laplace_smooth]
        class_0_mask_list_sum = laplace
        class_1_mask_list_sum = laplace

        for i in range(entry_num):
            if category_vector[i] == 1:
                class_1_mask_list_sum += entry_feature_list[i]
            else:
                class_0_mask_list_sum += entry_feature_list[i]
        class_0_mask_sum = np.sum(class_0_mask_list_sum)
        class_1_mask_sum = np.sum(class_1_mask_list_sum)
        self.p_word_0 = class_0_mask_list_sum / class_0_mask_sum
        self.p_word_1 = class_1_mask_list_sum / class_1_mask_sum
        return None

    # use test or val data
    def predict(self, entry_feature, with_log=False):
        entry = np.array(entry_feature)
        # 只保留entry存在的word的概率
        p_entry_feature_0 = self.p_word_0[entry > 0]
        p_entry_feature_1 = self.p_word_1[entry > 0]
        p_0_entry_feature, p_1_entry_feature = None, None

        if with_log == False:
            p_entry_feature_0 = np.log(p_entry_feature_0)
            p_entry_feature_1 = np.log(p_entry_feature_1)

            p_1_entry_feature = np.sum(p_entry_feature_1) + np.log(self.p_1_entry_feature)
            p_0_entry_feature = np.sum(p_entry_feature_0) + np.log(1.0 - self.p_1_entry_feature)
        else:
            p_1_entry_feature = reduce(lambda x1,x2: x1*x2, p_entry_feature_1) * self.p_1_entry_feature
            p_0_entry_feature = reduce(lambda x1,x2: x1*x2, p_entry_feature_0) * (1.0 - self.p_1_entry_feature)

        # info(p_0_entry_feature, p_1_entry_feature)
        return (0, 1)[p_1_entry_feature > p_0_entry_feature]

    pass

class native_bayes_sklearn(object):
    def __init__(self):
        self.classifier = MultinomialNB()
        return None

    # use training data
    @jit
    def fit(self, entry_feature_list, category_vector):
        self.classifier.fit(np.array(entry_feature_list), np.array(category_vector))
        return None

    # use test or val data
    @jit
    def predict(self, entry_feature):
        result = self.classifier.predict(np.array(entry_feature).reshape(1, -1))
        return result.item()

    # use test or val data
    @jit
    def score(self, entry_feature_list, category_vector):
        score_result = self.classifier.score(np.array(entry_feature_list), np.array(category_vector))
        return score_result.item()

    pass