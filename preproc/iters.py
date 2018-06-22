import torchtext
import os
import codecs
import jieba
from edit_distance import edit_distance
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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
DATA=os.path.join(HOME,'data_folder')

jieba.load_userdict(os.path.join(DATA,'dict.txt'))

def tokenizer(txt):
    return list(jieba.cut(txt))

def build_iters(ftrain='train.tsv',fvalid='valid.tsv',bsz=64):
    SEQ1 = torchtext.data.Field(sequential=True, tokenize=tokenizer,
                                pad_token=PAD_WORD, unk_token=UNK_WORD)
    SEQ2 = torchtext.data.Field(sequential=True, tokenize=tokenizer,
                                pad_token=PAD_WORD, unk_token=UNK_WORD)
    LBL = torchtext.data.Field(sequential=False, use_vocab=False)



    train, valid = torchtext.data.\
        TabularDataset.\
        splits(path=DATA, train=ftrain,
               validation=fvalid, format='tsv',
               fields=[
                   ('seq1', SEQ1),
                   ('seq2', SEQ2),
                   ('lbl', LBL)
               ])
    SEQ1.build_vocab(train)
    SEQ2.build_vocab(train)
    train_iter, val_iter = torchtext.data.Iterator.splits(
        (train, valid), batch_sizes=(bsz, bsz),
        device=-1, repeat=False, sort=False
    )

    return SEQ1, SEQ2,\
           train_iter, val_iter

def dataset_weight(train_iter):
    npos = 0
    nneg = 0
    for sample in train_iter:
        b_npos = sample.lbl.numpy().sum()
        npos += b_npos
        nneg += sample.lbl.shape[0]-b_npos

    return {'wpos': 1 - npos / (npos + nneg), 'wneg': 1 - nneg / (npos + nneg)}

def edistance_seg(train_iter):
    d_lst_pos = []
    d_lst_neg = []
    for sample in train_iter.dataset.examples:
        seq1, seq2, lbl  = sample.seq1, sample.seq2, sample.lbl
        d = edit_distance(seq1, seq2)
        if lbl == '1':
            d_lst_pos.append(d)
        else:
            d_lst_neg.append(d)

    draw_hist(d_lst_pos, 'Pos edistance', 'ed', '#',
              0, np.array(d_lst_pos).max() + 1, 0, 4000)
    draw_hist(d_lst_neg, 'Neg edistance', 'ed', '#',
              0, np.array(d_lst_neg).max() + 1, 0, 4000)

    return np.array(d_lst_pos).mean(), np.array(d_lst_pos).std(), \
           np.array(d_lst_neg).mean(), np.array(d_lst_neg).std()

def edistance_raw(data):
    d_lst_pos = []
    d_lst_neg = []

    with open(os.path.join(data,'train.tsv'),'r') as f:
        for line in f:
            seq1, seq2, lbl = line.strip().split('\t')
            d = edit_distance(list(seq1),list(seq2))
            if lbl == '1':
                d_lst_pos.append(d)
            else:
                d_lst_neg.append(d)

    draw_hist(d_lst_pos, 'Pos edistance', 'ed', '#',
              0, int(np.array(d_lst_pos).max()/2), 0, 4000)
    draw_hist(d_lst_neg, 'Neg edistance', 'ed', '#',
              0, int(np.array(d_lst_neg).max()/2), 0, 4000)

    return np.array(d_lst_pos).mean(), np.array(d_lst_pos).std(), \
           np.array(d_lst_neg).mean(), np.array(d_lst_neg).std()

def ed_classify_raw(data, threshold):
    pred_lst=[]
    true_lst=[]
    with open(os.path.join(data, 'train.tsv'), 'r') as f:
        for line in f:
            seq1, seq2, lbl = line.strip().split('\t')
            d = edit_distance(list(seq1), list(seq2))
            pred_lst.append(1 if d < threshold else 0)
            if lbl == '0' and d < threshold:
                print(seq1, seq2)
            true_lst.append(int(lbl))

    pred_lst = np.array(pred_lst)
    true_lst = np.array(true_lst)
    return {'threshold':threshold,
            'accur': metrics.accuracy_score(true_lst,pred_lst),
            'precision': metrics.precision_score(true_lst,pred_lst),
            'recall': metrics.recall_score(true_lst,pred_lst),
            'f1': metrics.f1_score(true_lst,pred_lst)}

def ed_classify(train_iter, threshold):
    pred_lst=[]
    true_lst=[]
    for sample in train_iter.dataset.examples:
        seq1, seq2, lbl = sample.seq1, sample.seq2, sample.lbl
        d = edit_distance(seq1, seq2)
        pred_lst.append(1 if d < threshold else 0)
        true_lst.append(int(lbl))

    pred_lst = np.array(pred_lst)
    true_lst = np.array(true_lst)
    return {'threshold':threshold,
            'accur': metrics.accuracy_score(true_lst,pred_lst),
            'precision': metrics.precision_score(true_lst,pred_lst),
            'recall': metrics.recall_score(true_lst,pred_lst),
            'f1': metrics.f1_score(true_lst,pred_lst)}


def draw_hist(data_lst,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(data_lst,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.xticks(range(Xmin,Xmax+1,2))
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()

# def data_augm_sogo(train_tsv, valid_tsv):
#     with open(os.path.join(DATA,'train_augm'),'w') as f:
#         with open(os.path.join(DATA,train_tsv),'r') as ftrain:
#             lines = ftrain.readlines()
#             for i, line in enumerate(lines):
#                 done = False
#
#                 while not done:
#                     try:
#                         seq1, seq2, lbl = line.split('\t')
#                         seq1_trans, seq2_trans = trans_sogo.trans_back(seq1), \
#                                                  trans_sogo.trans_back(seq2)
#                         f.write(seq1_trans+'\t'+seq2_trans+'\t'+lbl)
#
#                         done = True
#
#                         percent = i / len(lines)
#                         fill = int(percent * 40)
#                         print("\r[{}{}]: {:.4f} (train)".format("=" * fill, " " * (40 - fill), percent))
#
#                     except:
#                         print("\r[{}{}]: {:.4f} (train) error".format("=" * fill, " " * (40 - fill), percent))
#                         time.sleep(5)
#
#
#     print('\n')
#
#     with open(os.path.join(DATA,'valid_augm'),'w') as f:
#         with open(os.path.join(DATA,valid_tsv),'r') as fvalid:
#             lines = fvalid.readlines()
#             for i, line in enumerate(lines):
#
#                 done = False
#
#                 while not done:
#                     try:
#
#                         seq1, seq2, lbl = line.split('\t')
#                         seq1_trans, seq2_trans = trans_sogo.trans_back(seq1), \
#                                                  trans_sogo.trans_back(seq2)
#                         f.write(seq1_trans+'\t'+seq2_trans+'\t'+lbl)
#
#                         percent = i / len(lines)
#                         fill = int(percent * 40)
#                         print("\r[{}{}]: {:.4f} (valid)".format("=" * fill, " " * (40 - fill), percent))
#
#                     except:
#                         print("\r[{}{}]: {:.4f} (valid) error".format("=" * fill, " " * (40 - fill), percent))
#                         time.sleep(5)
#
# def data_augm_google(train_tsv, valid_tsv):
#     with open(os.path.join(DATA,'train_augm_google.txt'),'w') as f:
#         with open(os.path.join(DATA,train_tsv),'r') as ftrain:
#             lines = ftrain.readlines()
#             for i, line in enumerate(lines):
#                 done = False
#
#                 while not done:
#                     try:
#                         seq1, seq2, lbl = line.split('\t')
#                         seq1_trans, seq2_trans = google.trans_back(seq1), \
#                                                  google.trans_back(seq2)
#                         f.write(seq1_trans+'\t'+seq2_trans+'\t'+lbl)
#
#                         done = True
#
#                         percent = i / len(lines)
#                         fill = int(percent * 40)
#                         print("\r[{}{}]: {:.4f} (train)".format("=" * fill, " " * (40 - fill), percent))
#
#                     except:
#                         print("\r[{}{}]: {:.4f} (train) error".format("=" * fill, " " * (40 - fill), percent))
#                         time.sleep(5)
#
#
#     print('\n')
#
#     with open(os.path.join(DATA,'valid_augm_google.txt'),'w') as f:
#         with open(os.path.join(DATA,valid_tsv),'r') as fvalid:
#             lines = fvalid.readlines()
#             for i, line in enumerate(lines):
#
#                 done = False
#
#                 while not done:
#                     try:
#
#                         seq1, seq2, lbl = line.split('\t')
#                         seq1_trans, seq2_trans = google.trans_back(seq1), \
#                                                  google.trans_back(seq2)
#                         f.write(seq1_trans+'\t'+seq2_trans+'\t'+lbl)
#
#                         percent = i / len(lines)
#                         fill = int(percent * 40)
#                         print("\r[{}{}]: {:.4f} (valid)".format("=" * fill, " " * (40 - fill), percent))
#
#                     except:
#                         print("\r[{}{}]: {:.4f} (valid) error".format("=" * fill, " " * (40 - fill), percent))
#                         time.sleep(5)

if __name__ == '__main__':
    # SEQ1, SEQ2, \
    # train_iter, val_iter = build_iters(bsz=4)

    # print(dataset_weight(train_iter))
    #
    # print(edistance(train_iter))
    #
    # for threshold in range(1,15):
    #     print(ed_classify(train_iter,threshold))

    # data_augm_google('train.tsv','valid.tsv')
    # print(edistance_raw(data=DATA))
    # for threshold in range(1,15):
    #     print(ed_classify(DATA,threshold))
    print(ed_classify_raw(DATA, 4))

#
# train_iter, val_iter, test_iter = data.Iterator.splits(
#     (train, val, test), batch_sizes=(2,1,1),
#     device=-1, repeat=False
# )
#
# for sample in train_iter:
#     print(sample.Text, sample.Label)
