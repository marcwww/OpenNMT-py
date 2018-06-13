import torchtext
import os
import codecs
import jieba

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

if __name__ == '__main__':
    SEQ1, SEQ2, \
    train_iter, val_iter = build_iters(bsz=4)

    print(dataset_weight(train_iter))


#
#
# train_iter, val_iter, test_iter = data.Iterator.splits(
#     (train, val, test), batch_sizes=(2,1,1),
#     device=-1, repeat=False
# )
#
# for sample in train_iter:
#     print(sample.Text, sample.Label)
