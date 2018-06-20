import os
import codecs
import random

HOME=os.path.abspath('..')
DATA=os.path.join(HOME,'data')

def balance(data,data_sets):
    lines=[]
    for data_set in data_sets:
        lines.extend(codecs.open(os.path.join(data,data_set),'r',encoding='utf8').readlines())

    shuffled_idx = random.sample(list(range(len(lines))), k=len(lines))

    pos_idx=[]
    neg_idx=[]
    for idx in shuffled_idx:
        _, _ , lbl = lines[idx].strip().split('\t')
        if lbl == '1':
            pos_idx.append(idx)
        else:
            neg_idx.append(idx)

    npos=len(pos_idx)
    nneg=len(neg_idx)
    if npos < 0.5 * nneg:
        num = int(nneg/npos)
        for i in range(num):
            negs = neg_idx[i*npos:((i+1)*npos if i != num-1 else nneg)]
            train_i = [lines[idx] for idx in random.sample(negs+pos_idx, k=len(negs+pos_idx))]
            with open(os.path.join(data,'train%d.tsv' % (i)),'w') as f:
                f.writelines(train_i)

if __name__=='__main__':
    balance(DATA,['train.tsv'])
