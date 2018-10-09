# -*- encoding:utf-8 -*-
__author__ = 'SUTL'
"""
    加载数据
"""
import pickle
import numpy as np
from time import time
from collections import defaultdict
from util import read_lines


def load_word_embed(word_embed_dim=50):
    """
    加载word2vec模型
    Args:
        word_embed_dim: 词向量维度
    """
    assert word_embed_dim in [50, 100]
    path = './w2v_model/car_ner_w2v_model_%d.pk' \
        % word_embed_dim
    file_we = open(path,'rb')
    word_embed = pickle.load(file_we)
    file_we.close()
    return word_embed


def get_words_tags(sentence):
    """
    Args:
        sentence: a/n b/adj ...
    Return:
        words, tags
    """
    words, tags = [], []
    for item in sentence.split(' '):
        try:
            index = item.rindex('/')
        except Exception as e:
            continue
        word = item[:index]
        tag = item[index+1:]
        if word:
            words.append(word)
        if tag:
            tags.append(tag)
    return words, tags


def init_voc():
    """
    Initing vocabulary.
    Return:
        word_voc: 词表
        tag_voc: 词性表
        label_voc: label
        label_voc_rev: xx
    """
    lines = read_lines('./com_data/data_h/Train.csv')
    lines += read_lines('./com_data/data_h/Test.csv')
    # 单词->id, 词性标注->id, label->id
    word_dict = defaultdict(int)
    tag_set = []
    label_set = ['pos', 'neu', 'neg']
    for line in lines:
        sentence = ' '.join(line.split('|')[3:])
        words, tags = get_words_tags(sentence)
        tag_set += tags
        for word in words:
            word_dict[word] += 1
    # 排序
    word_dict = sorted(word_dict.items(),key=lambda d:d[0])
    word_voc = dict()
    for i, item in enumerate(word_dict):
        word_voc[item[0]] = i + 2  # start from 2
    tag_set = sorted(list(set(tag_set)))
    tag_voc = dict()
    for i, item in enumerate(tag_set):
        tag_voc[item] = i + 1  # start from 1
    label_voc = dict()
    label_set = sorted(label_set)
    for i, item in enumerate(label_set):
        label_voc[item] = i
    label_voc_rev = dict()  # 反转dict
    for item in label_voc.items():
        label_voc_rev[item[1]] = item[0]
    return word_voc, tag_voc, label_voc, label_voc_rev


def load_cilin():
    """
    加载同义词词林
    """
    path = './external_data/cilin/cilin.txt'
    lines = read_lines(path)
    cilin_dict = dict()
    for line in lines:
        items = line.split(' ')[1:]
        word_count = len(items)  # 同一类词的数量
        if word_count == 1:
            if items[0] in cilin_dict:
                continue
            cilin_dict[items[0]] = [items[0]]
        else:
            for i in range(len(items)):
                if items[i] in cilin_dict:
                    continue
                cilin_dict[items[i]] = items[:i] + items[i+1:]
    return cilin_dict


def get_syn_word(synset, w2v_model):
    """
    Args:
        synset: word的同义词集合
        w2v_model: 词向量模型
    Return:
        word的同义词，并且该同义词也在w2v_model，否则None
    """
    for word in synset:
        if word in w2v_model:
            return word
    return None


def init_word_weights(word_voc, word_embed, word_embed_dim=50):
    """
    初始化word_weights
    Args:
        word_embed: 预训练的词项量
        word_embed_dim:
    Return:
        word_weights
    """
    cilin_dict = load_cilin()  # 加载同义词词林
    word_voc_size = len(word_voc.items()) + 2
    word_weights = np.zeros((word_voc_size,word_embed_dim),dtype='float32')
    random_vec = np.random.uniform(-1,1,size=(word_embed_dim,))
    exist_count = 0  # 在词向量中存在的数量
    for word in word_voc:
        word_id = word_voc[word]
        if word in word_embed:  # 如果存在
            word_weights[word_id, :] = word_embed[word]
            exist_count += 1
        else:  # 若不存在，随机初始化 or 同义词替换
            if word in cilin_dict:  # 存在同义词
                # 获取word在词向量中存在的同义词
                syn_word = get_syn_word(cilin_dict[word], word_embed)
                if syn_word:
                    exist_count += 1
                    word_weights[word_id, :] = word_embed[syn_word]
            else:  # 随机初始化
                word_weights[word_id, :] = random_vec
    return word_weights


def init_position_weights(max_sent_len=100, position_embed_dim=5):
    """
    每个词到中心词的距离
    Args:
        max_sent_len: 最大句子长度
        position_embed_dim: 位子特征的维度
    Return:
        position_weights, np.array, float32
    """
    position_weights = np.random.uniform(-1, 1,
        (max_sent_len*2, position_embed_dim))
    position_weights[0, :] = 0.
    return position_weights


def init_tag_weights(tag_voc, max_sent_len=100, tag_embed_dim=50):
    """
    词性标记embedding
    Args:
        tag_voc: dict, 词性->id表
        max_sent_len: int, 最大句子长度
        tag_embed_dim: int, 词性维度
    Return:
        tag_weights: np.array, shape=[tag_voc_size, tag_embed_dim]
    """
    tag_voc_size = len(tag_voc.items()) + 1
    tag_weights = np.zeros((tag_voc_size,tag_embed_dim), dtype='float32')
    random_vec = np.random.uniform(-1,1,size=(tag_embed_dim,))
    path = './w2v_model/tags_50.pk'
    file = open(path, 'rb')
    tag_dict = pickle.load(file)
    file.close()
    for tag in tag_voc:
        tag_id = tag_voc[tag]
        if tag in tag_dict:
            tag_weights[tag_id, :] = tag_dict[tag]
        else:
            #print('error...', tag)
            tag_weights[tag_id, :] = random_vec
    return tag_weights


def get_sentence_ids(words, tags, word_voc, tag_voc,
                     target_index, max_sent_len):
    """
    词序列->词id序列
    Args:
        words:
        tags:
        word_voc:
        tag_voc:
    Return:
        xx
    """
    word_arr = np.zeros((max_sent_len,), dtype='int32')
    tag_arr = np.zeros((max_sent_len,), dtype='int32')
    position_arr = np.zeros((max_sent_len,), dtype='int32')
    shift = 0# max_sent_len - len(words)
    for i in range(len(words)):
        word = words[i]
        if word in word_voc:
            word_arr[i+shift] = word_voc[word]
        else:
            word_arr[i+shift] = 1  # 未登录词
    for i in range(len(tags)):
        tag = tags[i]
        if tag in tag_voc:
            tag_arr[i+shift] = tag_voc[tag]
        else:
            tag_arr[i+shift] = 1
    for i in range(len(words)):
        position_arr[i+shift] = i-target_index+max_sent_len
    return word_arr, tag_arr, position_arr


def get_sentiment_indices(tags_all):
    """
    获取情感词下标
    Args:
        tags_all;
    Return:
        indices: int list
    """
    indices = []
    tag_set = set(['pos', 'neg'])
    for i in range(len(tags_all)):
        if tags_all[i] in tag_set:
            indices.append(i)
    return indices


def find_target_accord2sl(target, words_all, tags_all):
    """
    若同一个view在一个句子出现多次，则根据target周围的情感词
    决定取哪一个view
    Args:
        target:
        words_all:
        tags_all:
    Return:
        xx
    """
    target_indices = []
    for i in range(len(words_all)):
        if target == words_all[i]:
            target_indices.append(i)
    if len(target_indices) == 0:  # 不存在view的情况
        return 0
    elif len(target_indices) == 1:  # 出现1次
        return target_indices[0]
    else:  # 出现多次的情况
        sentiment_indices = get_sentiment_indices(tags_all)  # 情感词下标
        if len(sentiment_indices) == 0:  # 没有情感词
            return target_indices[0]  # 默认取第一次出现的
        else:  # 计算与周围情感词的平均距离，取距离情感词们最近的
            distant = np.zeros(len(target_indices))  # 存放距离平均值
            sentiment_indices_arr = np.array(target_indices)
            for i in range(len(target_indices)):
                target_index = target_indices[i]
                distant[i] = sum(abs(sentiment_indices_arr-target_index)) / len(target_indices)
            return target_indices[distant.argmax()]


def cut_sentence(target, words_all, tags_all, max_sent_len):
    """
    Args:
        target:
        words_all:
        tags_all:
        max_sent_len:
    Return:
        xx
    """
    #if len(words_all) <= max_sent_len:
    #    return words_all, tags_all
    break_sign = ('。', '！', '？')  # 结束标记
    break_sign_2 = '，'  # 遇到两次停止
    target_index = -1  # target下标
    # 这里处理同一个view出现多次的情况
    target_index = find_target_accord2sl(target, words_all, tags_all)
    left_range = int(max_sent_len/2)
    right_range = max_sent_len-left_range-1
    start_index, end_index = target_index, target_index
    meet_max_time = 4
    # 向左扫描
    meet_time = 0
    for i in range(min(target_index, left_range)):
        start_index -= 1
        word = words_all[start_index]
        if word in break_sign:
            break
        if word == break_sign_2:
            meet_time += 1
            if meet_time >= meet_max_time:
                start_index += 1
                break
    # 向右扫描
    meet_time = 0
    for i in range(min(len(words_all)-target_index-1, right_range)):
        end_index += 1
        word = words_all[end_index]
        if words_all[end_index] in break_sign:
            break
        if word == break_sign_2:
            meet_time += 1
            if meet_time >= meet_max_time:
                break
    words = words_all[start_index:end_index+1]
    tags = tags_all[start_index:end_index+1]
    return words, tags


def load_train_data(word_voc, tag_voc, label_voc, max_sent_len=100, word_embed_dim=50,
		    sentence_len=False):
    """
    加载训练数据
    :param max_sent_len: 最大句子长度
    :param word_embed_dim: ...
    :return: xx
    """
    # 构造训练数据
    lines = read_lines('./com_data/data_h/Train.csv')
    train_count = len(lines)
    train_num = []
    train_targets_str = []  # targets
    train_target_indices = np.zeros((train_count, ), dtype='int32')
    train_sentence = np.zeros((train_count, max_sent_len), dtype='int32')  # sent
    train_sentence_len = np.zeros((train_count,), dtype='int32')  # 句子长度
    train_tag = np.zeros((train_count, max_sent_len), dtype='int32')  # tags
    train_position = np.zeros((train_count, max_sent_len), dtype='int32')  # target 在句子中的下标
    train_target = np.zeros((train_count, max_sent_len), dtype='int32')
    train_label = np.zeros((train_count,), dtype='int32')  # label
    for i in range(train_count):
        line = lines[i]
        items = line.split('|')
        num, target, label = items[:3] 
        label_id = label_voc[label]  # label id
        sentence_all = ' '.join(items[3:])
        words_all, tags_all = get_words_tags(sentence_all)
        words, tags = cut_sentence(target, words_all, tags_all, max_sent_len)  # sentence  截取
        train_sentence_len[i] = len(words)
        target_index = words.index(target) if target in words else 0  # target 在句子中的下标
        word_arr, tag_arr, position_arr = \
            get_sentence_ids(words,tags,word_voc,tag_voc,target_index,max_sent_len)

        train_num.append(num)
        train_targets_str.append(target)
        train_target_indices[i] = target_index  # new add 16-12-09
        train_sentence[i, :] = word_arr[:]
        train_tag[i, :] = tag_arr[:]
        train_position[i, :] = position_arr[:]
        if target in word_voc:
            train_target[i, :] = [0]*(max_sent_len-len(words)) + [word_voc[target]]*len(words)
        else:
            train_target[i, :] = [0] * max_sent_len
        train_label[i] = label_id
    train_data = [train_target_indices, train_sentence, train_tag, train_position, train_target, train_label, train_num, train_targets_str]
    if sentence_len:
        train_data.append(train_sentence_len)
    return train_data


def load_test_data(word_voc, tag_voc, label_voc, max_sent_len=100, word_embed_dim=50,
		   sentence_len=False):
    """
    加载测试数据
    """
    # 构造测试数据
    lines = read_lines('./com_data/data_h/Test.csv')
    test_count = len(lines)
    test_nums = []  # 测试编号
    test_targets_str = []  # targets
    test_target_indices = np.zeros((test_count, ), dtype='int32')
    test_sentence = np.zeros((test_count, max_sent_len), dtype='int32')  # sent
    test_sentence_len = np.zeros((test_count,), dtype='int32')  # 句子长度
    test_tag = np.zeros((test_count, max_sent_len), dtype='int32')  # tags
    test_position = np.zeros((test_count, max_sent_len), dtype='int32')  # target
    test_target = np.zeros((test_count, max_sent_len), dtype='int32')
    for i in range(test_count):
        line = lines[i]
        items = line.split('|')
        num, target = items[:2]
        sentence_all = ' '.join(items[2:])
        words_all, tags_all = get_words_tags(sentence_all)
        words, tags = cut_sentence(target, words_all, tags_all, max_sent_len)  # sentence  截取
        test_sentence_len[i] = len(words)
        target_index = words.index(target) if target in words else 0  # target 在句子中的下标
        word_arr, tag_arr, position_arr = \
            get_sentence_ids(words,tags,word_voc,tag_voc,target_index,max_sent_len)

        test_nums.append(num)
        test_targets_str.append(target)
        test_target_indices[i] = target_index
        test_sentence[i, :] = word_arr[:]
        test_tag[i, :] = tag_arr[:]
        test_position[i, :] = position_arr[:]
        if target in word_voc:
            test_target[i, :] = [0]*(max_sent_len-len(words)) + [word_voc[target]]*len(words)
        else:
            test_target[i, :] = [0] * max_sent_len
    test_data = [test_nums, test_targets_str, test_target_indices, test_sentence, test_tag, test_position, test_target]
    if sentence_len:
        test_data.append(test_sentence_len)
    return test_data


def load_data(max_sent_len=100, word_embed_dim=50, position_embed_dim=10,
              tag_embed_dim=50, sentence_len=False):
    """
    加载训练数据和测试数据，带情感词
    Args:
        max_sent_len: 句子最大长度
        word_embed_dim: 词向量维度(50 or 100)
        position_embed_dim: 位置向量维度
        tag_embed_dim: 词性向量维度
        sentence_len: 是否返回句子长度arr
    Returns:
        train_data, test_data, word_weights, position_weights, tag_weights, label_voc_rev
    """
    # 单词->id，词性->id
    word_voc, tag_voc, label_voc, label_voc_rev = init_voc()
    # word2vec model
    word_embed = load_word_embed(word_embed_dim=word_embed_dim)
    # word_weights
    word_weights = init_word_weights(word_voc, word_embed, word_embed_dim)
    # postion_weights
    position_weights = init_position_weights(max_sent_len, position_embed_dim)
    # word tag weights
    tag_weights = init_tag_weights(tag_voc, max_sent_len, tag_embed_dim)

    # 训练数据
    train_data = load_train_data(word_voc, tag_voc, label_voc, max_sent_len,
                     word_embed_dim, sentence_len=sentence_len)

    # 测试数据
    test_data = load_test_data(word_voc, tag_voc, label_voc, max_sent_len,
                    word_embed_dim, sentence_len=sentence_len)

    return train_data, test_data, word_weights, position_weights, tag_weights, label_voc_rev


def demo():
    t0 = time()
 
    train_data, test_data, word_weights, position_weights, tag_weights, label_voc_rev = load_data(100, 100, 50)

    test_nums, test_targets_str, test_target_indices, test_sentence, test_tag, test_position, test_target = test_data[:]
    print(test_nums[0])
    print(test_sentence[0])
    print(test_target_indices[:10])

    train_target_indices, train_sentence, train_tag, train_position, \
        train_target, train_label, train_num, train_targets_str = train_data[:]
    print(train_num[0])
    print(train_target_indices[:10])
    print(train_sentence[0])

    print('Done in %.1fs!' % (time()-t0))


if __name__ == '__main__': 
    demo()

