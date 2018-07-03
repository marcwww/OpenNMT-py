import torchtext
import os
import jieba
import re

from torchtext import data

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '<e>'
SEG_WORD = '<seg>'

# HOME=os.path.abspath('.')
HOME=os.path.abspath('..')
DATA=os.path.join(HOME,'data_folder')
STOP_WORDS=set()

with open(os.path.join(DATA, 'stop_words.txt'), 'r') as f:
    for line in f.readlines():
        w = line.split(' ')[0]
        STOP_WORDS.add(w)

jieba.load_userdict(os.path.join(DATA,'dict.txt'))

def tokenizer_word(txt):
    txt = re.sub('\*\*\*', '*', txt)
    return [w for w in jieba.cut(txt) if w not in STOP_WORDS]

def tokenizer_char(txt):
    # if txt.find('***') !=-1 :
    #     print('***')
    txt = re.sub('\*\*\*', '*', txt)
    res = [w for w in list(txt) if w not in STOP_WORDS]
    with open('res.txt','a+') as f:
        f.write(' '.join(res).encode('utf-8')+'\n')
    return res

def tokenizer_charNword(txt):
    res = []
    txt = re.sub('\*\*\*', '*', txt)
    for w in list(jieba.cut(txt)):
        if w in STOP_WORDS:
            continue
        res.append(w)
    return res

def build_iters(ftrain='train.tsv',fvalid='valid.tsv',bsz=64, level='char'):

    if level == 'word':
        tokenizer = tokenizer_word
    elif level == 'char':
        tokenizer = tokenizer_char
    else:
        tokenizer = tokenizer_charNword

    TEXT = torchtext.data.Field(sequential=True,
                                tokenize=tokenizer,
                                pad_token=PAD_WORD, unk_token=UNK_WORD)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    train = data.TabularDataset(path=os.path.join(DATA, ftrain),
                                    format='tsv',
                                    fields=[
                                        ('seq1', TEXT),
                                        ('seq2', TEXT),
                                        ('lbl', LABEL)
                                    ])
    TEXT.build_vocab(train)
    train_iter = data.Iterator(train, batch_size=bsz,
                               sort=False, repeat=False)

    valid = data.TabularDataset(path=os.path.join(DATA, fvalid), format='tsv',
                                fields=[
                                    ('seq1', TEXT),
                                    ('seq2', TEXT),
                                    ('lbl', LABEL)
                                ])

    valid_iter = data.Iterator(valid, batch_size=bsz,
                               sort=False, repeat=False)


    return TEXT, LABEL, train_iter, valid_iter

def build_iters_lm(ftrain='train.tsv',fvalid='valid.tsv',bsz=64, level='char'):

    if level == 'word':
        tokenizer = tokenizer_word
    elif level == 'char':
        tokenizer = tokenizer_char
    else:
        tokenizer = tokenizer_charNword

    TEXT = torchtext.data.Field(sequential=True,
                                tokenize=tokenizer,
                                pad_token=PAD_WORD,
                                unk_token=UNK_WORD,
                                eos_token=EOS_WORD)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    train = data.TabularDataset(path=os.path.join(DATA, ftrain),
                                    format='tsv',
                                    fields=[
                                        ('seq1', TEXT),
                                        ('seq2', TEXT),
                                        ('lbl', LABEL)
                                    ])
    TEXT.build_vocab(train)
    train_iter = data.Iterator(train, batch_size=bsz,
                               sort=False, repeat=False)

    valid = data.TabularDataset(path=os.path.join(DATA, fvalid), format='tsv',
                                fields=[
                                    ('seq1', TEXT),
                                    ('seq2', TEXT),
                                    ('lbl', LABEL)
                                ])

    valid_iter = data.Iterator(valid, batch_size=bsz,
                               sort=False, repeat=False)


    return TEXT, LABEL, train_iter, valid_iter

if __name__ == '__main__':
    TEXT, LABEL, train_iter, valid_iter = build_iters(level='char')
    # for sample in train_iter:
    #     print(sample.seq1)
