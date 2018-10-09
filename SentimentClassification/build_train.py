# -*- encoding:utf-8 -*-
__author__ = 'SUTL'
"""
    构造训练数据
    格式：
    编号|target|label|sentence
"""
import os
import re
from time import time
from util import read_lines
from nlpir import cut_sentence
from collections import defaultdict

def init_label_dict(path):
    """
    初始化训练数据
    :return: label_dict
    """
    lines = read_lines(path)
    label_dict = defaultdict(list)
    for line in lines[1:]:
        items = line.split('\t')
        num = items[0]
        car_name = re.sub('\s+','_',items[1])
        label = items[-1]
        label_dict[num].append((car_name,label))
    return label_dict

def format_words_tags(words_tags):
    """
    :param words_tags: '</w 凯越/car >/w'
    :return: words_tags
    """
    words = []
    for item in words_tags.split(' ')[1:-1]:
        index = item.rindex('/')
        words.append(item[:index])
    return ''.join(words) + '/car'

def find_indices(string, find):
    """
    :return: indices
    """
    end = False
    i = 0
    indices = []
    length = len(string)
    sub_length = len(find)
    while not end:
        sub_s = string[i:]
        if find not in sub_s:
            break
        if len(indices) > 0:
            index = indices[-1]+sub_length+sub_s.index(find)
        else:
            index = sub_s.index(find)
        indices.append(index)
        i = index + sub_length
        if i >= length:
            end = True
    return indices

def is_sign(num, sentence, index):
    """
    判断index周围是否有<>标记
    即index左边是否有<，或右边是否有>
    """
    # 向左扫描
    temp_index = index
    while temp_index >= 0:
        c = sentence[temp_index]
        if c == '<':  # 已标记
            return True
        if c == '>':  # 未标记
            return False
        temp_index -= 1
    temp_index = index
    while temp_index < len(sentence):
        c = sentence[temp_index]
        if c == '>':  # 已标记
            return True
        if c == '<':  # 未标记
            return False
        temp_index += 1
    return False

pattern_sub_1 = re.compile('<')
pattern_sub_2 = re.compile('>')
pattern_sub_3 = re.compile('\s+')
pattern_car = re.compile('(<.*?>/w)')
def build_train_data(label_dict, lines, file):
    """
    构造训练数据
    """
    error_count = 0
    error_num = []
    for line in lines[1:]:
        error = False
        items = line.split('\t')
        num = items[0]
        sentence = ' '.join(items[1:])	
        if not sentence.strip():
            continue
        sentence = pattern_sub_3.sub('_',sentence)
        sentence = pattern_sub_1.sub('',sentence)  # 删除'<','>'符号
        sentence = pattern_sub_2.sub('',sentence)
        labels = label_dict[num]
        # 标记出汽车名
        # 排序labels
        labels = sorted(labels, key=lambda d:len(d[0]), reverse=True)
        for name_label in labels:
            label = name_label[0]
            indices = find_indices(sentence, label)
            for index in indices[::-1]:
                if is_sign(num, sentence, index):  # 判断周围是否有<>标记
                    continue
                sentence_list = list(sentence)
                sentence_list[index:index+len(label)] = '<%s>'%label
                sentence = ''.join(sentence_list)
        words_tags = cut_sentence(sentence, string=True)  # str
        finds = pattern_car.findall(words_tags)
        for find in finds:
            find_sub = format_words_tags(find)
            words_tags = re.sub(find, find_sub, words_tags)
        for name_label in labels:
            name, label = name_label[:]
            if name not in sentence:
                error = True
            file.write('%s|%s|%s|%s\n' % (num, name, label, words_tags))
        if error:
            error_count += 1
            error_num.append(num)
    print(error_count)
    print(error_num)

def find_index_len(sentence):
    """
    Args:
        sentence: 词、词性组成的句子
    """
    words, tags = [], []
    for item in sentence.split(' '):
        if not item:
            continue
        index = item.rindex('/')
        word, tag = item[:index], item[index+1:]
        words.append(word)
        tags.append(tag)
    indices = []
    for i in range(len(tags)):
        if tags[i] == 'car':
            indices.append(i)
    index_len = []
    for index in indices:
        c_index = len(''.join(words[:index]))
        index_len.append((c_index, len(words[index])))
    return ''.join(words), index_len

if __name__ == '__main__':
    t0 = time()

    file = open('./com_data/data_h/Train.csv','w',encoding='utf-8')
    # 复赛数据
    label_dict = init_label_dict('./com_data/data_ori/Label.csv')
    lines = read_lines('./com_data/data_ori/Train.csv')
    build_train_data(label_dict, lines, file)
    
    # 初赛数据
    label_dict = init_label_dict('./com_data/data_ori/Label_first.csv')
    lines = read_lines('./com_data/data_ori/Train_first.csv')
    build_train_data(label_dict, lines, file)

    file.close()

    print('Done in %.1fs!' % (time()-t0))

