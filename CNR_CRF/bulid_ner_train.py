# -*- encoding:utf-8 -*-
"""
    构建CRF训练语料，2016-10-07
    
    CRF识别汽车名
"""
import os
import re
from time import time
from util import read_lines
from collections import defaultdict

def init_label_dict():
    """
    初始化训练数据
    :return: label_dict, {'num_1':[car_1,car_2,...], ...}
    """
    lines = read_lines('./data/Label.csv')
    label_dict = defaultdict(list)
    for line in lines[1:]:
        items = line.split('\t')
        num = items[0]
        car_name = re.sub('\s+','_',items[1])
        label = items[-1]
        label_dict[num].append((car_name,label))
    return label_dict

def find_indices(string, find):
    """
    找出find在string中的所有下标
    e.g.:
        find = 'sub'
        string = '123sub45sub6789sub012'
        则返回: [3, 8, 15]
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

pattern_sub_1 = re.compile('[<|>]')
pattern_sub_2 = re.compile('\s+')
pattern_car = re.compile('(<.*?>)')
def build_train_data():
    """
    构造训练数据
    """
    label_dict = init_label_dict()
    lines = read_lines('./data/Train.csv')
    file_crf = open('./crf/crf_train.txt','w',encoding='utf-8')
    newline_flag = ('。', '！', '？')
    for line in lines[1:]:
        if line[-1] not in newline_flag:
            line+='。'
        items = line.split('\t')
        num = items[0]
        sentence = ' '.join(items[1:])	
        if not sentence.strip():
            continue
        sentence = pattern_sub_2.sub('_',sentence)  # '_'替换空格等
        sentence = pattern_sub_1.sub('',sentence)  # 删除'<','>'符号
        labels = label_dict[num]
        # 标记出汽车名
        # 排序labels
        labels = sorted(labels, key=lambda d:len(d[0]), reverse=True)
        for name_label in labels:
            label = name_label[0]
            indices = find_indices(sentence, label)
            for index in indices[::-1]:
                if index-1>=0 and index+len(label)<len(sentence):
                    if sentence[index-1]=='<' or sentence[index+len(label)]=='>':  # if该部分已经标记
                        continue
                sentence_list = list(sentence)
                sentence_list[index:index+len(label)] = '<%s>'%label
                sentence = ''.join(sentence_list)  # e.g., "上市6年，<凯越>销量突破100万台。"
                
        # 删除'<|>'标记，并计算汽车名所在下标和每个汽车名长度
        indices_and_len = []  # [(index_1, len_1), ...]
        cars = pattern_car.findall(sentence)
        for car in cars:
            index = sentence.index(car)
            sentence = re.sub(car, car[1:-1], sentence, count=1)  # 除去'<|>'标记
            indices_and_len.append((index, len(car)-2))
        
        # 处理成CRF格式
        crf_tags = ['O'] * len(sentence)
        for item in indices_and_len:
            index, c_len = item[:]
            if c_len >= 3:
                crf_tags[index] = 'B-CAR'
                crf_tags[index+1:index+c_len-1] = ['I-CAR']*(c_len-2)
                crf_tags[index+c_len-1] = 'E-CAR'
            elif c_len == 2:
                crf_tags[index] = 'B-CAR'
                crf_tags[index+c_len-1] = 'E-CAR'
            else:  # 1
                crf_tags[index] = 'S-CAR'
        assert len(sentence) == len(crf_tags)
        for i in range(len(crf_tags)):
            c = sentence[i]
            file_crf.write('%s %s\n' % (c, crf_tags[i]))
            if c in newline_flag and i!=len(crf_tags)-1:
                file_crf.write('\n')
        file_crf.write('\n')
    file_crf.close()

#在训练语料中加入汽车词典这个特征，转移到其他类型的视角可替换成对应类型视角的字典
def Dic(presulr,aresult):
    fCar=open('dic/car_name.dic','r',encoding='utf-8')
    Car=[]
    CarLines=fCar.readlines()
    print(CarLines[0])
    for line in CarLines:
        if line.strip():
        word=line.strip()[:-4]
        #word=line.strip()
        Car.append(word)     
    fp=open(presulr,'r',encoding='utf-8')
    fa=open(aresult,'w',encoding='utf-8')
    line=fp.readline()
    word=[]
    dic = []
    gold=[]
    while line:
        if line.strip():
            w,g=line.strip().split()
            word.append(w)
            dic.append('O')
            gold.append(g)
        else:
            #print(word)
            #exit(0)
            i=0
            while i<len(word):
                start=i
                if i+10<len(word):
                    end=i+10
                else:
                    end=len(word)
                for j in range(end,start,-1):
                    if ''.join(word[i:j]) in Car:
                        if j-i==1:
                            dic[i]='S-CAR'
                        if j-i>1:
                            dic[i]='B-CAR'
                            for p in range(i+1,j-1):
                                dic[p]='I-CAR'
                            dic[j-1]='E-CAR'
                        break
                i=j

            for m in range(len(word)):
                fa.write(word[m]+'\t'+dic[m]+'\t'+gold[m]+'\n')
            fa.write('\n')
            word = []
            dic=[]
            gold = []
        line = fp.readline()

    fp.close()
    fa.close()

if __name__ == '__main__':
    t0 = time()
    build_train_data()
    Dic('crf/crf_train.txt','crf/Dic_train.txt')
    print('Done in %.1fs!' % (time()-t0))

