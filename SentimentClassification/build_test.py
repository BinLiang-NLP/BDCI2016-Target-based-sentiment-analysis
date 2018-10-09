# -*- encoding:utf-8 -*-
__author__ = 'SUTL'
"""
    构造测试数据
    格式：
    num|target|sentence
"""
import re
import json
from time import time
from nlpir import cut_sentence
from util import read_lines
from collections import defaultdict


def get_targets(write_str):
    """
    获取targets
    """
    items = write_str.split(' ')
    targets = []
    for item in items:
        if not item:
            continue
        index = item.rindex('/')
        name = item[:index]
        tag = item[index+1:]
        if tag == 'car':
            targets.append(name)
    return targets


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


pattern_sub = re.compile('\s+')
def init_test_sentence_dict():
    """
    xx
    """
    sentence_dict = dict()
    lines = read_lines('./com_data/data_ori/Test_nonoise.csv')
    for line in lines[1:]:
        items = line.split('\t')
        num = items[0]
        sentence = ' '.join(items[1:]).replace('|', '_')
        sentence = pattern_sub.sub('_', sentence)
        sentence_dict[num] = sentence
    return sentence_dict


def find_car_names(words_tags):
    """
    words_tags
    :return: xx
    """
    items = words_tags.split(' ')
    names = []
    for item in items:
        index = item.rindex('/')
        if 'car' == item[index+1:]:
            names.append(item[:index])
    return list(set(names))


pattern_sub_sign = re.compile('[<|>]')
pattern_car = re.compile('(<.*?>/w)')
pattern_errro = re.compile('[，|、|：|,|:|.|）|（|\(|\)|“|”]')
def build_test_data_from_crf():
    """
    根据crf识别结果构建测试语料
    """
    file_json = open('../CNR_CRF/temp/result_carDic.json','r')  # xx个
    json_list = json.load(file_json)
    file_json.close()
    file = open('./com_data/data_h/Test.csv','w',encoding='utf-8')
    file_view = open('./com_data/data_h/Test_view.csv','w',encoding='utf-8')
    file_crf_correct = open('./com_data/data_h/Test_crf_correct.csv','w',encoding='utf-8')
    sentence_dict = init_test_sentence_dict()
    count = 0
    correct_count = 0
    name_count = 0
    for json_dict in json_list:
        sentence_id = json_dict['SentenceID']
        sentence = pattern_sub_sign.sub('_', sentence_dict[sentence_id])
        sentence = list(sentence)  # 句子
        indices = json_dict['View']
        for index in indices[::-1]:
            if len(index) == 1:
                print(index)
                start, end = 0, index[0]
            else:
                start, end = index[:]
            sentence.insert(end+1, '>')
            sentence.insert(start, '<')
        sentence = ''.join(sentence)
        sentence_correct = sentence
        file_crf_correct.write('%s,%s\n' % (sentence_id, sentence_correct))
        file_view.write('%s,%s\n' % (sentence_id, sentence))
        words_tags = cut_sentence(sentence_correct, string=True)  # str 
        finds = pattern_car.findall(words_tags)
        for find in finds:
            find_sub = format_words_tags(find)
            index = find_sub.rindex('/')
            try:
                words_tags = re.sub(find, find_sub, words_tags, count=1)
            except Exception as e:
                print('Something wrong!')
        names = find_car_names(words_tags)
        for name in names:
            name_count += 1
            file.write('%s|%s|unknow|%s\n' % (sentence_id, name, words_tags))
    file.close()
    file_view.close()
    file_crf_correct.close()
    print(name_count, correct_count)


if __name__ == '__main__':
    t0 = time()

    build_test_data_from_crf()

    print('Done in %.1fs!' % (time()-t0))

