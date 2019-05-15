# -*- coding: UTF-8 -*-

#%% Compatible with jupyter
import os, sys

try:
    __ML_PATH__ = os.getcwd() + '/'
    __F_PATH__ = __ML_PATH__ + 'Bayes/exp_3/'
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
import matplotlib.pyplot as plt
import random
import jieba

"""
函数说明:中文文本处理
Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    glossary - 按词频降序排序的训练集列表
    train_entry_list - 训练集列表
    test_entry_list - 测试集列表
    train_category_vector - 训练集标签列表
    test_category_vector - 测试集标签列表
"""
@jit
def DataLoader(folder_path, val_rate=0.2):
    folder_list = os.listdir(folder_path)                           # 查看folder_path下的文件
    entry_list, category_vector = [], []

    for folder in folder_list:
        child_folder_path = os.path.join(folder_path, folder)       # 根据子文件夹，生成新的路径
        file_list = os.listdir(child_folder_path)                   # 存放子文件夹下的txt文件的列表

        # 遍历每个txt文件
        total_num = 0
        for file_name in file_list:
            if total_num > 100:                                     # 每类txt样本数最多100个
                break
            else:
                total_num += 1

            fp = open(os.path.join(child_folder_path, file_name), 'r', encoding='utf-8')
            raw_data = fp.read()
            fp.close()

            word_cut = jieba.cut(raw_data, cut_all=False)           # 精简模式，返回一个可迭代的generator
            entry = list(word_cut)                                  # generator转换为list

            entry_list.append(entry)                                # 添加数据集数据
            category_vector.append(folder)                            # 添加数据集类别

    glossary_dict = EntryList2Glossary(entry_list)
    sort_index = np.argsort(list(glossary_dict.values()))[::-1]     # argsort: reverse=False
    glossary = np.array(list(glossary_dict.keys()))[sort_index]
    val_num = int(len(entry_list) * val_rate)                       # 训练集和测试集切分的索引值
    zip_list = np.array(list(zip(entry_list, category_vector)))       # zip压缩合并，将数据与标签对应压缩
    np.random.shuffle(zip_list)                                     # 将category_vector 按行乱序
    test_entry_list, test_category_vector = zip_list[:val_num, 0], zip_list[:val_num, 1]
    train_entry_list, train_category_vector = zip_list[val_num:, 0], zip_list[val_num:, 1]

    return glossary, train_entry_list, test_entry_list, train_category_vector, test_category_vector

"""
函数说明:根据feature_words将文本向量化
Parameters:
    train_entry_list - 训练集
    test_entry_list - 测试集
    feature_list - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
"""
@jit
def TextFeatures(train_entry_list, test_entry_list, feature_list):
    train_feature_value_list, test_feature_value_list = [], []
    for entry in train_entry_list:
        train_feature_value_list.append(EntryEncode(feature_list, entry, 'sow'))
    for entry in test_entry_list:
        test_feature_value_list.append(EntryEncode(feature_list, entry, 'sow'))
    return train_feature_value_list, test_feature_value_list

"""
函数说明:文本特征选取\
Parameters:
    glossary - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集
"""
def Glossary2FeatureVec(glossary, stopwords_set, deleteN=0, feature_num=1000):
    feature_words = []
    n = 1
    for t in range(deleteN, len(glossary), 1):
        if n > feature_num:
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not glossary[t].isdigit() and glossary[t] not in stopwords_set and 1 < len(glossary[t]) < 5:
            feature_words.append(glossary[t])
        n += 1
    return feature_words

"""
函数说明:新闻分类器
Parameters:
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_category_vector - 训练集分类标签
    test_category_vector - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
"""
def nb_val(stopwords_set, glossary, train_entry_list, test_entry_list, train_category_vector, test_category_vector):
    classifier_sklearn = native_bayes_sklearn()
    deleteNs = range(0, 8500, 50)
    test_accuracy_list = []
    for deleteN in deleteNs:
        feature_words = Glossary2FeatureVec(glossary, stopwords_set, deleteN, 10000)
        train_feature_list, test_feature_list = TextFeatures(train_entry_list, test_entry_list, feature_words)
        classifier_sklearn.fit(train_feature_list, train_category_vector)
        test_accuracy = classifier_sklearn.score(test_feature_list, test_category_vector)
        test_accuracy_list.append(test_accuracy)
    return test_accuracy_list, deleteNs

if __name__ == '__main__':
    folder_path = __F_PATH__ + 'SogouC/Sample'                #训练集存放地址
    glossary, train_entry_list, test_entry_list, train_category_vector, test_category_vector = DataLoader(folder_path, 0.2)
    stopwords_file = __F_PATH__ + 'stopwords_cn.txt'
    stopwords_set = file2set(stopwords_file)

    test_accuracy_list, deleteNs = nb_val(stopwords_set, glossary, train_entry_list, test_entry_list, train_category_vector, test_category_vector)
    canvas, figure = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(5, 5))
    plt_draw2d(figure, deleteNs, test_accuracy_list, 'green', 'mix', u'current-learning-rate', u'current-learning-rate', u'deleteNs', u'test_accuracy')
    plt.show(canvas)