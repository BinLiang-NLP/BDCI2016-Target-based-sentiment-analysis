#-*- encoding:utf-8 -*-
__author__ = 'SUTL'
"""
    计算各项指标
"""
import nltk 
import numpy as np


def sim_compute(pro_labels, right_labels, ignore_label=None):
    """
    simple evaluate...
    :param pro_labels list : predict labels
    :param right_labels list : right labels
    :param ignore_label int : the label should be ignored
    :return acc, rec, f
    """
    assert len(pro_labels) == len(right_labels)
    acc_pro_labels, acc_right_labels = [], []
    rec_pro_labels, rec_right_labels = [], []
    labels_len = len(pro_labels)
    for i in range(labels_len):
        pro_label = pro_labels[i]
        if pro_label != ignore_label:  # 
            acc_pro_labels.append(pro_label)
            acc_right_labels.append(right_labels[i])
        if right_labels[i] != ignore_label:
            rec_pro_labels.append(pro_label)
            rec_right_labels.append(right_labels[i])
    acc_pro_labels, acc_right_labels = np.array(acc_pro_labels, dtype='int32'), \
        np.array(acc_right_labels, dtype='int32')
    rec_pro_labels, rec_right_labels = np.array(rec_pro_labels, dtype='int32'), \
        np.array(rec_right_labels, dtype='int32')
    acc = len(np.where(acc_pro_labels == acc_right_labels)[0]) / float(len(acc_pro_labels))
    rec = len(np.where(rec_pro_labels == rec_right_labels)[0]) / float(len(rec_pro_labels))
    f = (acc * rec * 2) / (acc + rec)
    return acc, rec, f


def get_max_label_len(classes_dict):
    """
    :param classes_dict
    """
    max_len = 0
    for item in classes_dict:
        label_name = classes_dict[item]
        if len(label_name) > max_len:
            max_len = len(label_name)
    return max_len


def compute(pro_labels, right_labels, ignore_label, classes_dict_rev, result_path):
    """
    with details
    :param pro_labels
    :param right_labels
    :param ignore_label
    :param classes_dict_rev
    :param result_path : the path to save result
    """
    classes_num = len(classes_dict_rev.items())
    each_class2others = np.zeros((classes_num, classes_num), dtype='int32')
    for i in range(len(right_labels)):
        right_label, pro_label = right_labels[i], pro_labels[i]
        each_class2others[right_label, pro_label] += 1
    max_class_len = get_max_label_len(classes_dict_rev)
    file_result = open(result_path, 'w', encoding='utf-8')
    # head
    file_result.write('%s' % (' '*(max_class_len+3)))
    for i in range(classes_num):
        file_result.write('%6s' % ('C-%d' % i))
    file_result.write('%6s%6s%6s%6s\n' % ('sum', 'acc', 'rec', 'f'))

    # compute score of each class
    each_acc, each_rec, each_f = np.zeros(classes_num, dtype='float32'), \
                                 np.zeros(classes_num, dtype='float32'), \
                                 np.zeros(classes_num, dtype='float32')
    for i in range(classes_num):
        each_acc[i] = each_class2others[i, i] / float(sum(each_class2others[:, i]))
        each_rec[i] = each_class2others[i, i] / float(sum(each_class2others[i, :]))
        each_f[i] = (each_acc[i] * each_rec[i] * 2) / (each_acc[i] + each_rec[i])
        
    # data
    for i in range(classes_num):
        file_result.write(('%'+str(max_class_len+1)+'s-%d') % (classes_dict_rev[i], i))
        for j in range(classes_num):
            file_result.write('%6d' % each_class2others[i, j])
        file_result.write(
            '%6d%6.2f%6.2f%6.2f\n' % 
            (sum(each_class2others[i]), each_acc[i]*100, each_rec[i]*100, each_f[i]*100)
	)
    
    # tail
    file_result.write(('%'+str(max_class_len+3)+'s') % 'sum')
    for i in range(classes_num):
        file_result.write('%6d' % sum(each_class2others[:, i]))
    acc, rec, f = sim_compute(pro_labels, right_labels, ignore_label)
    file_result.write(
        '%6d%6.2f%6.2f%6.2f' % 
	(np.sum(each_class2others), acc*100, rec*100, f*100)
    )
    file_result.close()
    print('\nresult has wrote to: %s' % result_path)
    return acc, rec, f


def demo():
    pro_labels =   [1,2,3,4,0,6,7,0,2,8]
    right_labels = [0,2,3,6,5,4,7,1,0,3]
    ignore_label = 0
    acc, rec, f = sim_compute(pro_labels, right_labels, ignore_label)
    print('acc:', acc)
    print('rec:', rec)
    print('  f:', f)


if __name__ == '__main__':
    demo()

