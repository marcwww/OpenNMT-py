# coding:utf-8
import os
import codecs
import random

HOME=os.path.abspath('..')
DATA=os.path.join(HOME,'data_folder')

def split(data,data_sets,ratio=0.9):
    lines=[]
    for data_set in data_sets:
        lines.extend(codecs.open(os.path.join(data,data_set),'r',encoding='utf8').readlines())

    num=len(lines)
    train_num=int(ratio*num)
    with open(os.path.join(data,'train.tsv'),'w') as f_train:
        with open(os.path.join(data,'valid.tsv'),'w') as f_test:
            for i in random.sample(list(range(num)),k=num):
                line = '\t'.join(lines[i].split('\t')[1:])
                if i<train_num:
                    f_train.write(line.encode('utf-8'))
                else:
                    f_test.write(line.encode('utf-8'))

if __name__=='__main__':
    split(DATA,['atec_nlp_sim_train.csv','atec_nlp_sim_train_add.csv'])
