import os
import codecs
import random

HOME=os.path.abspath('..')
data=os.path.join(HOME,'data')

def split(data_csv,ratio=0.7):
    lines=codecs.open(data_csv,'r',encoding='utf8').readlines()
    num=len(lines)
    train_num=int(ratio*num)
    with open(os.path.join(data,'train.tsv'),'w') as f_train:
        with open(os.path.join(data,'valid.tsv'),'w') as f_test:
            for i in random.sample(list(range(num)),k=num):
                if i<train_num:
                    f_train.write(lines[i])
                else:
                    f_test.write(lines[i])

if __name__=='__main__':
    split(os.path.join(data,'atec_nlp_sim_train.csv'))
