"""
根据获取的情感种子来扩展特定领域的情感词
"""

# -*- encoding:utf-8 -*-
from util import read_lines
from collections import defaultdict
from time import time 
import numpy as np
import extract_sentiment_seeds as ST
train_path = './com_data/data_h/Train.csv' # TODO
test_path = './com_data/data_h/Test_from_CRF.csv' # TODO
test_path_second = './com_data/data_h/Test_sencond.csv' # TODO
car_news_path = './external_data/DataGather/data_all/car_news_token.txt' # TODO
pos_seeds_path = './sentiment_words2.0/pos_seeds_second.txt'
neg_seeds_path = './sentiment_words2.0/neg_seeds_second.txt'
neu_seeds_path = './sentiment_words2.0/neu_seeds_second.txt'
without_neu_sentiment_path = './sentiment_words2.0/sentiment_words.txt'
syn_path = './sentiment_words/synonym.txt'
pos_words_path = './sentiment_words2.0/pos_words.txt'
neg_words_path = './sentiment_words2.0/neg_words.txt'
neu_words_path = './sentiment_words2.0/neu_words.txt'

def get_seeds(lines, theata):
    """
    将分数乘上一个比率
    """
    sentiment_seeds = dict()

    for line in lines:
        word = line.split(':')[0]
        score = int(line.split(':')[1])
        sentiment_seeds[word] = int(score * theata)
    return sentiment_seeds

def norm_seeds(pos_seeds, neg_seeds, neu_seeds):
    """
    去除冲突的种子
    """
    print('Before norm seeds length:', len(pos_seeds), len(neg_seeds), len(neu_seeds))
    temp = []
    for seed in pos_seeds:
        if seed in neg_seeds:
            neg_score = neg_seeds[seed]
            pos_score = pos_seeds[seed] 
            if pos_score>neg_score:
                del neg_seeds[seed]
            else:
                temp.append(seed)
        if seed in neu_seeds:
            neu_score = neu_seeds[seed] 
            pos_score = pos_seeds[seed]
            if pos_score>neu_score:
                del neu_seeds[seed]
            else:
                temp.append(seed)
    for seed in temp:
        del pos_seeds[seed]
    temp = []
    for seed in neg_seeds:
        if seed in neu_seeds:
            neu_score = neu_seeds[seed] 
            neg_score = neg_seeds[seed]
            if neg_score>neu_score:
                del neu_seeds[seed]
            else:
                temp.append(seed)
    for seed in temp:
        del neg_seeds[seed]
    print('After norm seeds length:', len(pos_seeds), len(neg_seeds), len(neu_seeds))
    # 去重后加上原来的seeds
    seeds1, seeds2, seeds3 = ST.load_sentiment_seeds()
    for seed in seeds1:
        if len(seed)<=1 or len(seed)>=5:
            continue
        if seed not in pos_seeds:
            pos_seeds[seed] = seeds1[seed]
    for seed in seeds2:
        if len(seed)<=1 or len(seed)>=5:
            continue
        if seed not in neg_seeds:
            neg_seeds[seed] = seeds2[seed]
    for seed in seeds3:
        if len(seed)<=1 or len(seed)>=5:
            continue
        if seed not in neu_seeds:
            neu_seeds[seed] = seeds3[seed]
    return pos_seeds, neg_seeds, neu_seeds

def load_sentiment_seeds():
    """
    加载情感种子
    """
    pos_lines = read_lines(pos_seeds_path)
    neg_lines = read_lines(neg_seeds_path)
    neu_lines = read_lines(neu_seeds_path)
    lines = pos_lines + neg_lines + neu_lines
    sentiment_seeds = get_seeds(lines, 1.0)
    pos_seeds = get_seeds(pos_lines, 0.32)
    neg_seeds = get_seeds(neg_lines, 1.0)
    neu_seeds = get_seeds(neu_lines, 0.13)
    pos_seeds, neg_seeds, neu_seeds = norm_seeds(pos_seeds, neg_seeds, neu_seeds)
    return sentiment_seeds, pos_seeds, neg_seeds, neu_seeds

def load_synonym():
    """
    分两种存储
    字典，键为同义词id，值为同义词列表
    字典，键为同义词单词，值为id
    """
    lines = read_lines(syn_path)
    synonym_words_id = dict()
    id_syn_words = dict()
    for line in lines:
        sy_id = line.split(' ')[0]
        words = line.split(' ')[1:]
        if len(words) <= 1:
            continue
        id_syn_words[sy_id] = words
        for word in words:
            synonym_words_id[word] = sy_id
    return synonym_words_id, id_syn_words


def load_data(sentiment_seeds):
    """
    加载测试集
    """
    lines = read_lines(test_path)
    lines += read_lines(test_path_second)
    candidate_sentiment_words = list()
    for line in lines:
        sentence = line.split('|')[3]
        for word_tag in sentence.split(' '):
            if '/' not in word_tag:
                continue
            word = word_tag.split('/')[0]
            if len(word) <= 1:
                continue
            tag = word_tag.split('/')[1]
            if tag == 'v' or tag == 'vi' or tag == 'd' or tag == 'a'\
                or tag == 'vn' or tag == 'pos' or tag == 'neg':
                    if word not in sentiment_seeds and word not in candidate_sentiment_words:
                        candidate_sentiment_words.append(word)
    return candidate_sentiment_words

def calculate_knn(pos_seeds, neg_seeds, neu_seeds, candidate_sentiment_words, synonym_words_id, id_syn_words):
    """
    计算knn的得分
    """

    sentiment_words = dict() # 键为word，值为极性
    without_neu_sentiment_words = dict()
    include_inver_words_sentiment = dict()
    pos_sen_words = list()
    neg_sen_words = list()
    neu_sen_words = list()
    labels = ['pos', 'neg', 'neu']
    count = [0, 0, 0] # 扩展得到的情感词
    all_count = [0, 0, 0] # 扩展得到的情感词加上情感种子

    for word in candidate_sentiment_words:
        if word not in synonym_words_id:
            continue
        # syn_words = []
        # syn_words.append(word)
        knn_score = [0, 0, 0]
        sy_id = synonym_words_id[word]
        syn_words = id_syn_words[sy_id]
        for w in syn_words:
            if w in pos_seeds:
                knn_score[0] += pos_seeds[w] # TODO... 改为加1
            if w in neg_seeds:
                knn_score[1] += neg_seeds[w]
            if w in neu_seeds:
                knn_score[2] += 1
        if max(knn_score) == 0:
            continue
        label_id = knn_score.index(max(knn_score))
        label = labels[label_id]
        if label_id != 2:
            without_neu_sentiment_words[word] = label
        if label_id == 0:
            pos_sen_words.append(word)
        elif label_id == 1:
            neg_sen_words.append(word)
        sentiment_words[word] = label
        count[label_id] += 1
    print(count)
    all_count = count
    for word in pos_seeds:
        pos_sen_words.append(word)
        sentiment_words[word] = labels[0]
        without_neu_sentiment_words[word] = labels[0]
        all_count[0] += 1
    for word in neg_seeds:
        neg_sen_words.append(word)
        sentiment_words[word] = labels[1]
        without_neu_sentiment_words[word] = labels[1]
        all_count[1] += 1
    for word in neu_seeds:
        sentiment_words[word] = labels[2]
        all_count[2] += 1
    print(all_count)
    print('len pos neg:', len(pos_sen_words), len(neg_sen_words))
    return without_neu_sentiment_words

def save_sentiment_words(without_neu_sentiment_words):
    """

    """
    without_neu_sentiment_words = sorted(without_neu_sentiment_words.items(), key=lambda a:len(a[0]), reverse=True)
    with open(without_neu_sentiment_path, 'w') as fp:
        for item in without_neu_sentiment_words:
            string = item[0] + ' ' + item[1] + '\n'
            fp.write(string)

if __name__ == '__main__':
    sentiment_seeds, pos_seeds, neg_seeds, neu_seeds = load_sentiment_seeds()
    synonym_words_id, id_syn_words = load_synonym()
    candidate_sentiment_words = load_data(sentiment_seeds)
    without_neu_sentiment_words = calculate_knn(pos_seeds, neg_seeds, neu_seeds,
            candidate_sentiment_words, synonym_words_id, id_syn_words)
    save_sentiment_words(without_neu_sentiment_words)
    print(len(without_neu_sentiment_words))
    print('Done!')
