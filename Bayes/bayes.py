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

"""
函数说明: 朴素贝叶斯分类器训练函数
Parameters:
    data_mask_list - 训练文档矩阵，即Entry2GlossaryMask返回的glossary_mask构成的矩阵
    class_vector - 训练类别标签向量，即DataLoader返回的class_vector
Returns:
    p_word_0 - 对类别为1的所有entry, 统计glossary中的每个单词类出现的概率
    p_word_1 - 对类别为0的所有entry, 统计glossary中的每个单词类出现的概率
    p_1_entry - entry_list中词条类别为1的概率
"""
class native_bayes(object):
    """
        to deal with the 0 probability problem
        using Laplace smoothing to quantized input(>=0) to (0.0, 1.0]
    """
    @jit
    def __init__(self, data_mask_list, class_vector, laplace_smooth=True):
        entry_num = len(data_mask_list)
        self.p_1_entry = sum(class_vector)/float(entry_num)

        laplace = (0, 1)[laplace_smooth]
        class_0_mask_list_sum = laplace
        class_1_mask_list_sum = laplace

        for i in range(entry_num):
            if class_vector[i] == 1:
                class_1_mask_list_sum += data_mask_list[i]
            else:
                class_0_mask_list_sum += data_mask_list[i]
        class_0_mask_sum = sum(class_0_mask_list_sum)
        class_1_mask_sum = sum(class_1_mask_list_sum)
        self.p_word_0 = class_0_mask_list_sum / class_0_mask_sum
        self.p_word_1 = class_1_mask_list_sum / class_1_mask_sum
        return None

    def classify(self, entry, with_log = False):
        # # 只保留entry存在的word的概率
        p_entry_mask_0 = self.p_word_0[entry > 0]
        p_entry_mask_1 = self.p_word_1[entry > 0]

        if with_log == False:
            p_entry_mask_0 = np.log(p_entry_mask_0)
            p_entry_mask_1 = np.log(p_entry_mask_1)

            p_1_entry = sum(p_entry_mask_1) + np.log(self.p_1_entry)
            p_0_entry = sum(p_entry_mask_0) + np.log(1.0 - self.p_1_entry)
        else:
            p_1_entry = reduce(lambda x1,x2: x1*x2, p_entry_mask_1) * self.p_1_entry
            p_0_entry = reduce(lambda x1,x2: x1*x2, p_entry_mask_0) * (1.0 - self.p_1_entry)

        # info(p_0_entry, p_1_entry)
        return (0, 1)[p_1_entry > p_0_entry]

pass