# -*- encoding:utf-8 -*-
import os
import re
import subprocess
import codecs
from time import time


pattern_sub= re.compile('\s+')
#在crf测试语料中加入汽车字典这个特征，转移到其他类型的视角可替换成对应类型视角的字典
def carDictest(presulr,aresult):
    fCar=open('./dict/car_name.dic','r',encoding='utf-8')
    Car=[]
    CarLines=fCar.readlines()
    #print(CarLines[0])
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
    #gold=[]
    while line:
        if line.strip():
            w=line.strip()
            word.append(w)
            dic.append('O')
            #gold.append(g)
        else:
            #print(word)
            #exit(0)
            i=0
            while i<len(word):
                start=i
                if i+10<len(word):#长度从10开始匹配，若字典中没有这个词，则从第九个匹配，依次递减。
                    end=i+10
                else:
                    end=len(word)
                pos=i+1
                for j in range(end,start,-1):
                    if ''.join(word[i:j]) in Car:
                        if j-i==1:
                            dic[i]='S-CAR'
                        if j-i>1:
                            dic[i]='B-CAR'
                            for p in range(i+1,j-1):
                                dic[p]='I-CAR'
                            dic[j-1]='E-CAR'
                        pos=j
                        break
                i=pos
            for m in range(len(word)):
                fa.write(word[m]+'\t'+dic[m]+'\n')
            fa.write('\n')
            word = []
            dic=[]
            #gold = []
        line = fp.readline()
    fp.close()
    fa.close()


#用crf预测视角
def crf_prodict(testName,resultName):
    crf_test_exe = r'crf_test '
    model = r' -m model/Dic_model'
    test=r' '+testName
    output =r' '+resultName
    process = subprocess.Popen(crf_test_exe + model +test + ' >' + output, shell=True)
    process.wait()  # 堵塞式


#将预测的结果文件转化为列表
def toList(resultName):
    fin=open(resultName,'r',encoding='utf-8')
    ids=fid.readlines()
    lines=fin.readlines()
    for line in lines:
        if line.strip():
            tag=line.strip().split()[2]
            Tlist.append(tag)
    #print(Tlist)
    view=[]
    SE = []
    for i in range(len(Tlist)):
        if Tlist[0] == 'S':
            SE.append(i)
            SE.append(i)
            view.append(SE,Tlist[2:])
            #index += 1
            SE = []
        if Tlist[0] == 'B':
            SE.append(i)
        if Tlist[i] == 'E':
            SE.append(i)
            view.append(SE,Tlist[2:])
            if len(SE)==1:
                print(id)
                print(SE)
                print(Tlist)
            #index+=1
            SE = []
    fin.close()
    return view


#识别一段文本的视角
def recognize_NE(sentence):
    fout=open('temp/test.txt','w',encoding='utf-8')
    sentence = pattern_sub.sub('_',content)  # '_'替换空格等
    #将需要识别视角的文本转化成crf测试语料的格式
    for x in sentence:
        fout.write(x+'\n')
        if x in ['。','！','？']:
            fout.write('\n')
    fout.write('\n')
    carDictest('temp/test.txt','temp/test_car.txt')
    crf_prodict('temp/test_car.txt','temp/result_car.txt')
    view=toList('temp/result_car.txt')
    print(view)
    return view


#批量识别文本的视角
def recognize_NE_batch(sentences):
    viewList=[]
    if sentences:
        for sentence in sentences:
            viewList+=recognize_NE(sentence)
    print(viewList)
    return viewList


pattern_sub= re.compile('\s+')
def build_test_data():
    file=codecs.open('data/Test.csv','r','utf-8')
    fid=codecs.open('idOfTest','w','utf-8')
    path='temp/test/'
    line=file.readline()
    line=file.readline()
    while line:
        if len(line.strip().split())!=2:
            #print(line)
            id=line.strip().split()[0]
            content=' '.join(line.strip().split()[1:])
        else:
            id,content=line.strip().split()
        fid.write(id+'\n')
        fout=open(path+id,'w',encoding='utf-8')

        sentence = pattern_sub.sub('_',content)  # '_'替换空格等
        
        for x in sentence:
            fout.write(x+'\n')
            if x in ['。','！','？']:
                fout.write('\n')
        fout.write('\n')
        fout.close()

        line = file.readline()


def crf_prodict_car():
    crf_test_exe = r'crf_test '
    model = r' -m ./model/Dic_model'
    root = './temp/test_carDic/'
    paths = os.listdir(root)
    #进行测试
    for path in paths:
        print(path)
        test=r' temp/test_carDic/'+path
        output =r' temp/result_carDic/'+path
        process = subprocess.Popen(crf_test_exe + model +test + ' >' + output, shell=True)
        process.wait()  # 堵塞式


def toJson():
    fid=open('idOfTest','r',encoding='utf-8')
    fout=open('temp/result_carDic.json','w',encoding='utf-8')
    ids=fid.readlines()
    number=0
    index=0
    fout.write('[')
    for id in ids:
        id=id.strip()
        fin=open('temp/result_carDic/'+id,'r',encoding='utf-8')
        Tlist=[]
        view=[]
        lines=fin.readlines()
        for line in lines:
            if line.strip():
                tag=line.strip().split()[2]
                Tlist.append(tag)
        #print(Tlist)
        SE = []
        for i in range(len(Tlist)):
            if Tlist[i] == 'S-CAR':
                SE.append(i)
                SE.append(i)
                view.append(SE)
                #index += 1
                SE = []
            if Tlist[i] == 'B-CAR':
                SE.append(i)
            if Tlist[i] == 'E-CAR':
                SE.append(i)
                view.append(SE)
                if len(SE)==1:
                    print(id)
                    print(SE)
                    print(Tlist)
                #index+=1
                SE = []
        #print(view)
        index+=len(view)
        if len(view)==0:
            continue
        if id==ids[-1].strip():
            fout.write('{"SentenceID":' + '"%s"' % id + ',"View":%s' % str(view) + '}')
        else:
            fout.write('{"SentenceID":'+'"%s"'%id+',"View":%s'%str(view)+'},')
    fout.write(']')
    print(index)
    fid.close()
    fout.close()


if __name__ == '__main__':

    t0 = time()

    #识别一段文本的视角
    #recognize_NE('虽然官方并没有推出什么特别的防弹版本，但雷克萨斯的offroad向来都是私人装甲的热门车型。')
    #批量识别文本的视角
    #configure_NER_batch('') #参数为文本列表
    build_test_data()

    root = './temp/test/'
    paths = os.listdir(root)
    for path in paths:
        te1='./temp/test/'+path
        te2='./temp/test_carDic/'+path
        carDictest(te1,te2)

    crf_prodict_car()

    toJson()  # 保存为json格式的文件

    print('Done in %.1fs!' % (time()-t0))


