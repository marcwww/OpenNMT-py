import torchtext
from torchtext import data
import os
import codecs
import jieba
from edit_distance import edit_distance
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
# import trans_sogo
# import google
import time

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'

global HOME
HOME=os.path.abspath('.')
# HOME=os.path.abspath('..')
DATA=os.path.join(HOME,'data')

jieba.load_userdict(os.path.join(DATA,'dict.txt'))

def tokenizer(txt):
    return list(jieba.cut(txt))

def build_iters(ftrains,ftrain_all='train.tsv',fvalid='valid.tsv',bsz=64):
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer,
                                pad_token=PAD_WORD, unk_token=UNK_WORD)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    train_all = data.TabularDataset(path=os.path.join(DATA,ftrain_all),
                            format='tsv',
                            fields=[
                                ('seq1',TEXT),
                                ('seq2',TEXT),
                                ('lbl',LABEL)
                            ])
    TEXT.build_vocab(train_all)

    train_iters = []
    for ftrain in ftrains:
        train = data.TabularDataset(path=os.path.join(DATA,ftrain),
                            format='tsv',
                            fields=[
                                ('seq1',TEXT),
                                ('seq2',TEXT),
                                ('lbl',LABEL)
                            ])
        train_iter = data.Iterator(train,batch_size=bsz,
                                   sort=False,repeat=False)
        train_iters.append(train_iter)

    valid = data.TabularDataset(path=os.path.join(DATA,fvalid),format='tsv',
                                fields=[
                                    ('seq1',TEXT),
                                    ('seq2',TEXT),
                                    ('lbl',LABEL)
                                ])
    valid_iter = data.Iterator(valid,batch_size=bsz,
                               sort=False,repeat=False)

    return TEXT,train_iters,valid_iter
