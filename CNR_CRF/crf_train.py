# -*- encoding:utf-8 -*-
import subprocess
crf_train_exe = r'./crf_tool/crf_learn'
template = r' ./crf/template2'
train=r' ./crf/Dic_train.txt'
model =r' ./model/Dic_model'
#训练模型
process = subprocess.Popen(crf_train_exe + template + train + model, shell=True)
process.wait()  # 堵塞式

