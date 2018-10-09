# -*- encoding:utf-8 -*-
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
nb_epoch = 7
nb_filter = 200
nb_hiddens = 300
word_embed_dim = 100  # 100 or 300
position_embed_dim = 10   # v0.2新增
tag_embed_dim = 50  # v0.4新增
nb_classes = 3
max_sent_len = 100
#class_embbed_dim = 250  # 关系类别embedding维度 att-based pool

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
    load_data(max_sent_len, word_embed_dim, position_embed_dim, tag_embed_dim, sentence_len=False)
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
#print(train_label.shape, len(np.where(train_label == 1)[0]))
#print(dev_label.shape, len(np.where(dev_label == 1)[0]))
train_label = np_utils.to_categorical(train_label, nb_classes)
dev_label = np_utils.to_categorical(dev_label, nb_classes)

def max_1d(X):
    return K.max(X, axis=1)

def min_1d(X):
    return K.min(X, axis=1)

# Input attention
def get_att(X, index):
    result, update = theano.scan(lambda v, u: T.dot(v, T.transpose(u)), sequences=X, non_sequences=X[index])
    result_soft = T.nnet.softmax(result)  # T.exp(result) / T.sum(T.exp(result))
    A = T.diag(T.flatten(result_soft))  # 对角阵，n×n
    return T.dot(A, X)

def get_input_att(Xs, target_indices):
    """
    :param X: 输入
    :param target_index: target在句子中的下标
    :return: xx
    """
    result, update = theano.scan(lambda X, index: get_att(X, index), sequences=[Xs, target_indices])
    return result 
# end input attention

# 构建模型
print('Building model...')
act = LeakyReLU()
#input_target_indices = Input(shape=(1,), dtype='int32', name='input_target')
input_sentence = Input(shape=(max_sent_len,), dtype='int32', name='input_sentence')
embed_sentence = Embedding(output_dim=word_embed_dim, input_dim=word_weights.shape[0],
                           input_length=max_sent_len, weights=[word_weights],
                           dropout=0.12, name='embed_sentence')(input_sentence)
input_tag = Input(shape=(max_sent_len,), dtype='int32', name='input_sentiment')
embed_tag = Embedding(output_dim=tag_embed_dim, input_dim=tag_weights.shape[0],
                      input_length=max_sent_len, weights=[tag_weights],
                      dropout=0.12, name='embed_tag')(input_tag)
input_position = Input(shape=(max_sent_len,), dtype='int32', name='input_position')
embed_position = Embedding(output_dim=position_embed_dim, input_dim=position_weights.shape[0],
                           input_length=max_sent_len, weights=[position_weights],
                           dropout=0.12, name='embed_position')(input_position)
X_sentence = merge([embed_sentence, embed_position, embed_tag],
                   mode='concat', concat_axis=2)
# input attention
#X_sentence_att = Lambda(get_input_att, arguments={'target_indices': input_target_indices}, name='X_sentence_att')(X_sentence)

cnn_layer_2 = Convolution1D(nb_filter=nb_filter, filter_length=2,  # 窗口2
    border_mode='valid', activation='relu', name='conv_window_2')(X_sentence)
pool_output_2 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_2)

cnn_layer_3 = Convolution1D(nb_filter=nb_filter, filter_length=3,  # 窗口3
    border_mode='valid', activation='relu', name='conv_window_3')(X_sentence)
pool_output_3 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_3)

cnn_layer_4 = Convolution1D(nb_filter=nb_filter, filter_length=4,  # 窗口4
    border_mode='valid', activation='relu', name='conv_window_4')(X_sentence)
pool_output_4 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_4)

cnn_layer_5= Convolution1D(nb_filter=nb_filter, filter_length=5,  # 窗口5
    border_mode='valid', activation='relu', name='conv_window_5')(X_sentence)
pool_output_5 = Lambda(max_1d, output_shape=(nb_filter,))(cnn_layer_5)

pool_output = merge([pool_output_2, pool_output_3, pool_output_4, pool_output_5],
    mode='concat', name='pool_output')
X_dropout = Dropout(0.5)(pool_output)
X_output = Dense(nb_classes,
    W_constraint=maxnorm(3),
    W_regularizer=l2(0.01), 
    activity_regularizer=activity_l2(0.01),
    activation='softmax')(X_dropout)

model = Model(input=[input_sentence, input_position, input_tag], output=[X_output])
model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()
#exit()
print('Train...')
model_path = './model/best_model.hdf5'
modelcheckpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
model.fit([train_sentence, train_position, train_tag],
          [train_label],
          validation_data=([dev_sentence, dev_position, dev_tag], [dev_label]),
          callbacks=[modelcheckpoint],
          nb_epoch=nb_epoch, batch_size=batch_size)

# 保存模型
#print('Save model...')
#model_path = './model/'
#if not os.path.exists(model_path):
#    os.mkdir(model_path)
#model.save(model_path+'multi-window_cnn.h5')
#del model

# 重新加载模型
#model = load_model('./model/best_model_100.hdf5')

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

