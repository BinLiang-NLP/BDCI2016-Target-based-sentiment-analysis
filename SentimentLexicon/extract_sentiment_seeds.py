"""
    通过普通情感词获取训练集中能作为情感种子的情感词
    把情感词储存到文件，供扩展情感词使用
"""
# -*- encoding:utf-8 -*-
from util import read_lines
from collections import defaultdict
from time import time 
import numpy as np

pos_seeds_path = './sentiment_words/posdict.txt'
neg_seeds_path = './sentiment_words/negdict.txt'
neu_seeds_path = './sentiment_words/middict.txt'
train_path = './com_data/data_h/Train.csv'
test_path = './com_data/data_h/Test_from_CRF.csv'
train_path_second = './com_data/data_h/Train_second.csv'
test_path_second = './com_data/data_h/Test_sencond.csv'
pos_seeds_path_write = './sentiment_words2.0/pos_seeds_second.txt'
neg_seeds_path_write = './sentiment_words2.0/neg_seeds_second.txt'
neu_seeds_path_write = './sentiment_words2.0/neu_seeds_second.txt'
sentiment_seeds_path = './sentiment_words2.0/sentiment_seeds.txt'

def load_train_data():
    """
    加载训练集
    将训练集分为pos,neg,neu
    将句子的word,tag存入data
    """
    lines = read_lines(train_path)
    lines += read_lines(train_path_second)
    pos_data = list() # 保存数据的word_tag，元素为列表，即一个句子[[]]
    neg_data = list()
    neu_data = list()

    for line in lines:
        try:
            num = line.split('|')[0]
            pol = line.split('|')[2]
            sentence = line.split('|')[3]
        except:
            print(num)
            continue
        words = []
        if pol == 'pos':
            for word_tag in sentence.split(' '):
                if '/' in word_tag:
                    words.append(word_tag)
            pos_data.append(words)
        elif pol == 'neg':
            for word_tag in sentence.split(' '):
                if '/' in word_tag:
                    words.append(word_tag)
            neg_data.append(words)
        elif pol == 'neu':
            for word_tag in sentence.split(' '):
                if '/' in word_tag:
                    words.append(word_tag)
            neu_data.append(words)
    print('length od pos data:', len(pos_data))
    print('length of neg data:', len(neg_data))
    print('length of neu data:', len(neu_data))
    return pos_data, neg_data, neu_data

def load_sentiment_seeds():
    """
    加载情感种子(网上搜集的情感集合)
    把情感种子存放到字典中，键为情感种子，值为情感种子得分
    """
    pos_lines = read_lines(pos_seeds_path)
    neg_lines = read_lines(neg_seeds_path)
    neu_lines = read_lines(neu_seeds_path)

    pos_seeds = dict()
    neg_seeds = dict()
    neu_seeds = dict()

    for line in pos_lines:
       if len(line.split(',')) > 2: # 多个,的情况忽略
            continue
       seed, score = line.split(',')[0], line.split(',')[1]
       pos_seeds[seed] = int(score)
    for line in neg_lines:
        if len(line.split(',')) > 2:
            continue
        seed, score = line.split(',')[0], line.split(',')[1]
        neg_seeds[seed] = int(score)
    for line in neu_lines:
        if len(line.split(',')) > 2:
            continue
        seed, score = line.split(',')[0], line.split(',')[1]
        neu_seeds[seed] = int(score)

    print(len(pos_seeds), len(neg_seeds), len(neu_seeds))
    
    temp = []
    # 去除冲突的情感词
    for i in pos_seeds:
        flag = 0
        if i in neg_seeds:
            flag = 1
            del neg_seeds[i]
        if i in neu_seeds:
            flag = 1
            del neu_seeds[i]
        if flag:
            temp.append(i)
    for i in temp:
        del pos_seeds[i]
    temp = []
    for i in neg_seeds:
        if i in neu_seeds:
            temp.append(i)
            del neu_seeds[i]
    for i in temp:
        del neg_seeds[i]
    print(len(pos_seeds), len(neg_seeds), len(neu_seeds))
    return pos_seeds, neg_seeds, neu_seeds

def generate_sentiment_seeds(data, seeds, words_tags):
    """
    对3个数据集分别统计情感种子
    """
    # words_tags = dict() # 键为单词，值为词性
    words_tf = defaultdict(int) # 键为单词，值为单词在数据集中出现的次数
    # sen_candidate = list() # 保存候选情感集
    for line in data:
        for word_tag in line:
            word = word_tag.split('/')[0]
            if len(word)<=1:
                continue
            tag = word_tag.split('/')[1]
            # if flag:
            if word in seeds:
                words_tf[word] += seeds[word]
            elif 'a' == tag or 'v' == tag or 'vi' == tag or 'd' == tag:
                if len(word) > 1:
                    words_tf[word] += 1
            if word not in words_tags:
                words_tags[word] = tag
    for word in words_tf:
        if word in seeds:
            score = seeds[word]
            if score<=1:
                continue
            # words_tf[word] *= score # TODO ...乘以score或者不乘
    t_words_tf = words_tf # 保存字典
    words_tf = sorted(words_tf.items(), key = lambda a:a[1], reverse=True)
    return words_tf, words_tags, t_words_tf

def save_sentiment_seeds(pos_words_tf, neg_words_tf, neu_words_tf, 
        pos_words_dict, neg_words_dict, neu_words_dict,
        pos_data, neg_data, neu_data):
    """
    保存每个数据集中的top n作为情感种子
    """
    with open(pos_seeds_path_write, 'w') as fp:
        for item in pos_words_tf:
            compare = 0
            if item[0] in neg_words_dict:
                compare += neg_words_dict[item[0]]
            if item[0] in neu_words_dict:
                compare += neu_words_dict[item[0]]
            theata = compare / ((len(neg_data) + len(neu_data)) / len(pos_data))
            # print('pos theata:', theata)
            if item[1] > theata:
                string = item[0] + ':' + str(item[1]) + '\n'
                # print(string.strip())
                fp.write(string)
    with open(neg_seeds_path_write, 'w') as fp:
        for item in neg_words_tf:
            compare = 0
            # compare2 = 0
            if item[0] in pos_words_dict:
                compare += pos_words_dict[item[0]]
            if item[0] in neu_words_dict:
                compare += neu_words_dict[item[0]]
            theata = compare / ((len(pos_data) + len(neu_data)) / len(neg_data))
            # print('neg theata:', theata)
            # theata1 = compare1 * 0.3
            # theata2 = compare2 * 0.1
            if item[1] > theata:
                string = item[0] + ':' + str(item[1]) + '\n'
                # print(string.strip())
                fp.write(string)
    with open(neu_seeds_path_write, 'w') as fp:
        for item in neu_words_tf:
            compare = 0
            if item[0] in pos_words_dict:
                compare += pos_words_dict[item[0]]
            if item[0] in neg_words_dict:
                compare += neg_words_dict[item[0]]
            theata = compare / ((len(pos_data) + len(neg_data)) / len(neu_data))
            # print('neu theata:', theata)
            if item[1] > theata:
                string = item[0] + ':' + str(item[1]) + '\n'
                fp.write(string)

if __name__ == '__main__':
    pos_data, neg_data, neu_data = load_train_data()
    print(len(pos_data), len(neg_data), len(neu_data))
    pos_seeds, neg_seeds, neu_seeds = load_sentiment_seeds()

    words_tags = dict()
    print('words_tags:', len(words_tags))
    pos_words_tf, words_tags, pos_words_dict = generate_sentiment_seeds(pos_data, pos_seeds, words_tags)
    print('words_tags', len(words_tags))
    neg_words_tf, words_tags, neg_words_dict = generate_sentiment_seeds(neg_data, neg_seeds, words_tags)
    print('words_tags', len(words_tags))
    neu_words_tf, words_tags, neu_words_dict = generate_sentiment_seeds(neu_data, neu_seeds, words_tags)

    # 保存情感种子（Top N）
    # save_sentiment_seeds(pos_words_tf, pos_seeds_path_write)
    # save_sentiment_seeds(neg_words_tf, neg_seeds_path_write)
    save_sentiment_seeds(pos_words_tf, neg_words_tf, neu_words_tf,
            pos_words_dict, neg_words_dict, neu_words_dict,
            pos_data, neg_data, neu_data)

    print('Done!')



