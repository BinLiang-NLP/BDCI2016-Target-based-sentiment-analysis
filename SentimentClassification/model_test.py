# -*- encoding:utf-8 -*-
__author__ = 'SUTL'
"""
    sc_model v0.4, 2016-10-09

    0.642
"""
import os
import re
from collections import defaultdict
from util import read_lines
from time import time
import numpy as np
from dataset import load_data
from keras.layers import Input, Embedding, Activation, \
    LSTM, GRU, Convolution1D, Lambda, Dense, Dropout, merge
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, \
    PReLU, ELU, ParametricSoftplus, ThresholdedReLU, SReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adamax
from keras.constraints import maxnorm
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from evaluate import sim_compute, compute
import theano
import theano.tensor as T
from theano import scan
from theano import pp

# 参数设置
batch_size = 32
nb_epoch = 20
nb_lstm = 1000
nb_filter = 200
nb_hiddens = 300
word_embed_dim = 100  # 100 or 300
position_embed_dim = 10   # v0.2新增
tag_embed_dim = 50  # v0.4新增
nb_classes = 3
max_sent_len = 100
class_embbed_dim = 55  # 关系类别embedding维度

def init_sentence_id():
    """
    初始化待提交sentence id
    """
    sentence_id = set()
    lines = read_lines('./com_data/data_ori/Test.csv')
    for line in lines[1:]:
        items = line.split('\t')
        num = items[0]
        sentence_id.add(num)
    return sentence_id

# 加载数据
print('Loading data...')
train_data, test_data, word_weights, position_weights, tag_weights, label_voc_rev = \
    load_data(max_sent_len, word_embed_dim, position_embed_dim, tag_embed_dim, sentence_len=False, ex_data=False)
td_target_indices, td_sentence, td_tag, td_position, td_target, td_label, td_num, td_target_str = train_data[:]
seed = 12345
np.random.seed(seed)
np.random.shuffle(td_target_indices)
np.random.seed(seed)
np.random.shuffle(td_sentence)
np.random.seed(seed)
np.random.shuffle(td_tag)
np.random.seed(seed)
np.random.shuffle(td_position)
np.random.seed(seed)
np.random.shuffle(td_target) 
np.random.seed(seed)
np.random.shuffle(td_label)
np.random.seed(seed)
np.random.shuffle(td_num)
np.random.seed(seed)
np.random.shuffle(td_target_str)
test_nums, test_targets_str, test_target_indices, test_sentence, test_tag, test_position, test_target = test_data[:]
#ex_nums, ex_targets_str, ex_sentence, ex_tag, ex_position, ex_target = ex_data[:]
# 划分train, dev
boundary = int(len(td_label) / 5.0)
dev_target_indices, dev_sentence, dev_position, dev_tag, dev_label = \
    td_target_indices[:boundary], td_sentence[:boundary], td_position[:boundary], td_tag[:boundary], td_label[:boundary]
train_target_indices, train_sentence, train_position, train_tag, train_label = \
    td_target_indices[boundary:], td_sentence[boundary:], td_position[boundary:], td_tag[boundary:], td_label[boundary:]
train_label = np_utils.to_categorical(train_label, nb_classes)
dev_label = np_utils.to_categorical(dev_label, nb_classes)

# 重新加载模型
model = load_model('./model/best_model_300.hdf5')

print('Predict...')
pre = model.predict([test_sentence, test_position, test_tag])
pre_labels = []
for p in pre:
    pre_labels.append(p.argmax())
# 写入文件
result_path = './result'
if not os.path.exists(result_path):
    os.mkdir(result_path)
file = open(result_path+'/result_bs.csv','w',encoding='utf-8')
sentence_id = init_sentence_id()
file.write('SentenceId,View,Opinion\n')
for i in range(len(pre_labels)):
    label = label_voc_rev[pre_labels[i]]
    target = re.sub('_', ' ', test_targets_str[i])
    num = test_nums[i]
    if num in sentence_id:
        sentence_id.remove(num)
    file.write('%s,%s,%s\n' % (num, target.strip(), label))
#for s_id in sentence_id:
#    file.write('%s,,\n' % s_id)
file.close()

print('Done!')

