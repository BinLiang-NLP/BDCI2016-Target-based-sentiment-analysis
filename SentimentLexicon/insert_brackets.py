# -*- encoding:utf-8 -*-
"""
    加载数据

    max_len取100，不足的补全，过长的在句子两边截取。

    训练集测试集词表大小：31004
"""
import pickle
import numpy as np
from time import time
from collections import defaultdict
from util import read_lines
import re
target_count = dict()
w_path = './com_data/data_h/cut_target_count.txt'
train_miss = []
test_miss = []

def load_word_embed(word_embed_dim=50):
    """
    加载word2vec模型
    """
    #path = './w2v_model/car_w2v_model_100.pk'
    # path = './w2v_model/car_w2v_100_sent_second.pk'
    path = './w2v_model/include_kuohao_car_sentiment_100.pk'
    file_we = open(path,'rb')
    word_embed = pickle.load(file_we)
    file_we.close()
    return word_embed

def load_embedding(word_embed_dim=50):
    """
    word embedding
    other embedding...
    :return: word_embed, other_embedding...
    """
    word_embed = load_word_embed(word_embed_dim=word_embed_dim)
    return word_embed

def get_words_tags(sentence):
    """
    :param sentence: a/n b/adj ...
    :return: words, tags
    """
    words, tags = [], []
    for item in sentence.split(' '):
        try:
            index = item.rindex('/')
        except Exception as e:
            continue
        word = item[:index]
        tag = item[index+1:]
        words.append(word)
        tags.append(tag)
    return words, tags

def init_voc():
    """
    Initing vocabulary.
    return:
        word_voc: 词表
        tag_voc: 词性表
        label_voc: label
    """
    tags_list = ['neg', 'pos', 'a', 'car', 'p', 'd', 'vn']
    have = 0
    lines = read_lines('./com_data/data_h/Train1.csv')
    lines += read_lines('./com_data/data_h/Test1.csv')
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
    word_dict = sorted(word_dict.items(),key=lambda d:d[1], reverse=True)
    word_voc = dict()
    word_voc['<'] = 2
    word_voc['>'] = 3
    for i, item in enumerate(word_dict):
        if item[0] in word_voc:
            continue
        word_voc[item[0]] = i + 4  # 单词下标从2开始
    tag_set = sorted(list(set(tag_set)))
    tag_voc = dict()
    for i, item in enumerate(tag_set):
        have += 1
        tag_voc[item] = i + 1  # 词性下标从1开始
    label_voc = dict()
    label_set = sorted(label_set)
    for i, item in enumerate(label_set):
        label_voc[item] = i
    label_voc_rev = dict()  # 反转dict
    for item in label_voc.items():
        label_voc_rev[item[1]] = item[0]
    print('have:', have)
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

def get_syn_word(synset, word_embed):
    """
    :param synset: word的同义词集合
    :param word_embed: 词向量模型
    """
    for word in synset:
        if word in word_embed:
            return word
    return None

def init_word_weights(word_voc, word_embed, word_embed_dim=50):
    """
    初始化word_weights
    :param word_embed: 预训练的词项量
    :param word_embed_dim:
    :return: word_weights
    """
    cccc = 0
    m_cccc = 0
    cilin_dict = load_cilin()  # 加载同义词词林
    word_voc_size = len(word_voc.items()) + 4
    word_weights = np.zeros((word_voc_size,word_embed_dim),dtype='float32')
    random_vec = np.random.uniform(-0.5,0.5,size=(word_embed_dim,))
    exist_count = 0  # 在词向量中存在的数量
    for word in word_voc:
        # if '_' in word:
        # t_word = re.sub(' ', '_', word)
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
            else:  # 不存在同义词 TODO 转换成字向量加和？
                if '_' in word:
                    print(word)
                m_cccc += 1
                word_weights[word_id, :] = random_vec
                    #np.random.uniform(-1,1,size=(word_embed_dim,))
    #print(word_voc_size, exist_count)
    #exit()
    print('m_cccc:', m_cccc)
    return word_weights

def init_position_weights(max_sent_len=100, position_embed_dim=5):
    """
    每个词到中心词的距离
    :param max_sent_len: 最大句子长度
    :param position_embed_dim: 位子特征的维度
    :return: position_weights, np.array, float32
    """
    position_weights = np.random.uniform(-1, 1,
        (max_sent_len*2, position_embed_dim))
    position_weights[0, :] = 0.
    return position_weights


def init_tag_weights(tag_voc, max_sent_len=100, tag_embed_dim=5):
    """
    词性标记embedding
    """
    tag_embed_count = 0
    path = './w2v_model/tags_without50.pk'
    file_we = open(path,'rb')
    tag_embed = pickle.load(file_we)
    file_we.close()
    tag_voc_size = len(tag_voc.items()) + 1
    tag_weights = np.random.uniform(-1, 1, (tag_voc_size, tag_embed_dim))
    tag_weights[0, :] = 0.
    for tag in tag_voc:
        tag_id = tag_voc[tag]
        if tag in tag_embed:
            # tag_weights[tag_id, :] = tag_embed[tag]
            tag_embed_count += 1
    print('tag_embed_count:', tag_embed_count)
    return tag_weights

def get_sentence_ids(words, tags, word_voc, tag_voc,
                     target_index, max_sent_len):
    """
    词序列->词id序列
    :param words:
    :param tags:
    :param word_voc:
    :param tag_voc:
    :return: xx
    """
    # max_sent_len += 2
    word_arr = np.zeros((max_sent_len,), dtype='int32')
    tag_arr = np.zeros((max_sent_len,), dtype='int32')
    position_arr = np.zeros((max_sent_len,), dtype='int32')
    shift = max_sent_len - len(words)
    shift = 0
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
    """
    indices = []
    tag_set = set(['pos', 'neg'])
    for i in range(len(tags_all)):
        if tags_all[i] in tag_set:
            indices.append(i)
    return indices

# add by ljx, 2016-11-12
def find_target_accord2sl(target, words_all, tags_all):
    """
    若同一个view在一个句子出现多次，则根据target周围的情感词
    决定取哪一个view
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
    :param target:
    :param words_all:
    :param tags_all:
    :param max_sent_len:
    :return: xx
    """
    #if len(words_all) <= max_sent_len:
    #    return words_all, tags_all
    break_sign = ('。', '！', '？')  # 结束标记
    break_sign_2 = '，'  # 遇到两次停止
    target_index = -1  # target下标
    # 这里处理同一个view出现多次的情况
    target_index = find_target_accord2sl(target, words_all, tags_all)
    #for i in range(len(words_all)):
    #    if words_all[i] == target:
    #        target_index = i
    #if target_index==-1:
    #    #print('No target exists!')
    #    # exit()
    #    target_index = 0
    max_sent_len -= 2
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

def get_persentence_target_count(num, target, words, words_all, en=None):
    """
    获取每个句子中target出现的次数
    """
    count = 0
    flag = 1
    for word in words:
        if word == target:
            flag = 0
            count += 1
    if count in target_count:
        target_count[count] += 1
    else:
        target_count[count] = 1
    if flag:
        t = num + '|' + target
        for word in words:
            if word in target:
                t += '-' + word
        if en == 'train':
            train_miss.append(t)
        elif en == 'test':
            test_miss.append(t)
        # for word in words_all:
            # if word == target:
                # train_miss.append(t)
                # break


def load_train_data(word_voc, tag_voc, label_voc, max_sent_len=100, word_embed_dim=50,
		    sentence_len=False):
    """
    加载训练数据
    :param max_sent_len: 最大句子长度
    :param word_embed_dim: ...
    :return: xx
    """
    # TODO 构造训练数据
    lines = read_lines('./com_data/data_h/Train1.csv')
    train_count = len(lines)
    train_num = []
    train_targets_str = []  # targets
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
        #print(line)
        if len(words_all)>max_sent_len - 2:
            words, tags = cut_sentence(target, words_all, tags_all, max_sent_len)  # sentence  截取
        else:
            words, tags = words_all, tags_all
        get_persentence_target_count(num, target, words, words_all, en='train')
        train_sentence_len[i] = len(words)
        target_index = words.index(target)+1 if target in words else 0  # target 在句子中的下标
        if target in words: # 标志target的位置，如<宝马>
            words.insert(words.index(target), '<')
            words.insert(words.index(target) + 2, '>')
            tags.insert(words.index(target),'wcar') # 词性标为wcar
            tags.insert(words.index(target)+2,'wcar')
        word_arr, tag_arr, position_arr = \
            get_sentence_ids(words,tags,word_voc,tag_voc,target_index,max_sent_len)
        # xx
        train_num.append(num)
        train_targets_str.append(target)
        train_sentence[i, :] = word_arr[:]
        train_tag[i, :] = tag_arr[:]
        train_position[i, :] = position_arr[:]
        if target in word_voc:
            train_target[i, :] = [0]*(max_sent_len-len(words)) + [word_voc[target]]*len(words)
        else:
            train_target[i, :] = [0] * max_sent_len
        train_label[i] = label_id
    train_data = [train_sentence, train_tag, train_position, train_target, train_label, train_num, train_targets_str]
    if sentence_len:
        train_data.append(train_sentence_len)
    return train_data


def load_test_data(word_voc, tag_voc, label_voc, max_sent_len=100, word_embed_dim=50,
		   sentence_len=False):
    """
    加载测试数据
    """
    # 构造测试数据
    w_fp = open('./com_data/data_h/guize1.csv', 'w', encoding='utf-8')
    lines = read_lines('./com_data/data_h/Test1.csv')
    test_count = len(lines)
    test_nums = []  # 测试编号
    test_targets_str = []  # targets
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
        if pol != '':
            t_target = re.sub('_', ' ', target)
            string1 = num + ',' + t_target + ',' + pol + '\n'
            w_fp.write(string1)
        if len(words_all)>max_sent_len - 2:
            words, tags = cut_sentence(target, words_all, tags_all, max_sent_len)  # sentence  截取
        else:
            words, tags = words_all, tags_all
        get_persentence_target_count(num, target, words, words_all, en='test')
        test_sentence_len[i] = len(words)
        target_index = words.index(target)+1 if target in words else 0  # target 在句子中的下标
        if target in words: # 标志target的位置，如<宝马>
            words.insert(words.index(target), '<')
            words.insert(words.index(target) + 2, '>')
            tags.insert(words.index(target),'wcar') # 词性标为wcar
            tags.insert(words.index(target)+2,'wcar')
        word_arr, tag_arr, position_arr = \
            get_sentence_ids(words,tags,word_voc,tag_voc,target_index,max_sent_len)
        # xx
        test_nums.append(num)
        test_targets_str.append(target)
        test_sentence[i, :] = word_arr[:]
        test_tag[i, :] = tag_arr[:]
        test_position[i, :] = position_arr[:]
        if target in word_voc:
            test_target[i, :] = [0]*(max_sent_len-len(words)) + [word_voc[target]]*len(words)
        else:
            test_target[i, :] = [0] * max_sent_len
    w_fp.close()
    test_data = [test_nums, test_targets_str, test_sentence, test_tag, test_position, test_target]
    if sentence_len:
        test_data.append(test_sentence_len)
    return test_data

def init_sentiment_weights(sentiment_embed_dim=5):
    """
    初始化sentiment_weights
    :params sentiment_embed_dim: 情感词强度的维度
    :return: xx
    """
    lines = read_lines('./external_data/sentiment.csv')
    sentiment_weight_dict = dict()
    for line in lines[1:]:
        items = line.split(',')
        word = items[0]
        sent_type = items[4]  # 情感词类型
        sent_weight = int(items[6])  # 情感词强度
        if sent_type.startswith('N'):  # 消极
            sentiment_weight_dict[word] = sent_weight + 11
        else:  # 积极
            sentiment_weight_dict[word] = sent_weight + 1
    # 初始化embedding
    sentiment_weights = np.random.uniform(-1,1,(21,sentiment_embed_dim))
    sentiment_weights[0, :] = 0.
    return sentiment_weight_dict, sentiment_weights

def load_data(max_sent_len=100, word_embed_dim=50, position_embed_dim=5,
              tag_embed_dim=50, sentiment_embed_dim=5, sentence_len=False, ex_data=False):
    """
    加载训练数据和测试数据，带情感词
    """
    # 单词->id，词性->id
    word_voc, tag_voc, label_voc, label_voc_rev = init_voc()
    # word2vec model, TODO other...
    word_embed = load_embedding(word_embed_dim=word_embed_dim)
    # 构造word_weights
    word_weights = init_word_weights(word_voc, word_embed, word_embed_dim)
    # 构造postion_weights
    position_weights = init_position_weights(max_sent_len, position_embed_dim)
    # 构造word tag weights
    tag_weights = init_tag_weights(tag_voc, max_sent_len, tag_embed_dim)

    # 训练数据
    train_data = load_train_data(word_voc, tag_voc, label_voc, max_sent_len,
                     word_embed_dim, sentence_len=sentence_len)

    # 测试数据
    test_data = load_test_data(word_voc, tag_voc, label_voc, max_sent_len,
                    word_embed_dim, sentence_len=sentence_len)

    # 外部数据
    if ex_data:
        external_data = load_external_data(word_voc, tag_voc, label_voc,
            max_sent_len=100, word_embed_dim=50)

    if ex_data:
        return train_data, test_data, external_data, word_weights, position_weights, tag_weights, label_voc_rev
    else:
        return train_data, test_data, word_weights, position_weights, tag_weights, label_voc_rev

def load_external_data(word_voc, tag_voc, label_voc, max_sent_len=100, word_embed_dim=50):
    """
    加载外部数据
    """
    lines = read_lines('./external_train_data/external_train.csv')
    test_count = len(lines)
    test_nums = []  # 测试编号
    test_targets_str = []  # targets
    test_sentence = np.zeros((test_count, max_sent_len), dtype='int32')  # sent
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
        target_index = words.index(target)+1 if target in words else 0  # target 在句子中的下标
        word_arr, tag_arr, position_arr = \
            get_sentence_ids(words,tags,word_voc,tag_voc,target_index,max_sent_len)
        # xx
        test_nums.append(num)
        test_targets_str.append(target)
        test_sentence[i, :] = word_arr[:]
        test_tag[i, :] = tag_arr[:]
        test_position[i, :] = position_arr[:]
        if target in word_voc:
            test_target[i, :] = [0]*(max_sent_len-len(words)) + [word_voc[target]]*len(words)
        else:
            test_target[i, :] = [0] * max_sent_len
    test_data = [test_nums, test_targets_str, test_sentence, test_tag, test_position, test_target]
    return test_data

if __name__ == '__main__': 
    t0 = time()
 
    train_data, test_data, word_weights, position_weights, tag_weights, label_voc_rev = load_data(100, 100, 5, ex_data=False)

    test_nums, test_targets_str, test_sentence, test_tag, test_position, test_target = test_data[:]
    print(test_sentence.shape)

    print('Done in %.1fs!' % (time()-t0))

    target_count = sorted(target_count.items(), key = lambda a:a[1], reverse = True)
    with open(w_path, 'w') as fp:
        for item in target_count:
            string = str(item[0]) + ':' + str(item[1]) + '\n'
            fp.write(string)
    print('done!')
    print('-------------------train----------------')
    for i in train_miss:
        print(i)
    print('-------------------test----------------')
    for i in test_miss:
        print(i)


