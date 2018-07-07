#coding:utf-8
import torchtext
import os
import jieba
import re
import langconv

from torchtext import data

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '<e>'
SEG_WORD = '<seg>'

HOME=os.path.abspath('.')
# HOME=os.path.abspath('..')
DATA=os.path.join(HOME,'data_folder')
STOP_WORDS=set()

file_res = open('res.txt', 'w')

with open(os.path.join(DATA, 'stop_words.txt'), 'r') as f:
    for line in f.readlines():
        w = line.split(' ')[0]
        STOP_WORDS.add(w)

jieba.load_userdict(os.path.join(DATA,'dict.txt'))

def tokenizer_word(txt):
    txt = re.sub('\*\*\*', '*', txt)
    return [w for w in jieba.cut(txt) if w not in STOP_WORDS]

def tokenizer_char(txt):

    def seg_zh(matched):
        begin, end = matched.regs[0]
        phrase = matched.string[begin:end]
        # phrase_simp = phrase
        phrase_simp = langconv.Converter('zh-hans').\
            convert(phrase)
        return ' '+' '.join(list(phrase_simp))+' '

    def match_en(matched):
        begin, end = matched.regs[0]
        word = matched.string[begin:end]
        if len(word)>1:
            return ' '+word+' '
        else:
            return ' '

    def match_symbol(matched):
        begin, end = matched.regs[0]
        symbols = matched.string[begin:end]
        return ' '+symbols[0]+' '

    txt = re.sub(u'\*+', ' * ', txt)
    txt = re.sub(u'[a-zA-z]+', match_en, txt)
    txt = re.sub(u'[\u4e00-\u9fa5]+', seg_zh, txt)
    txt = re.sub(u'[^ a-zA-Z\u4e00-\u9fa5\*]+', match_symbol, txt)
    txt = re.sub('\s+', ' ', txt)
    res = txt.split(' ')
    file_res.write((' '.join(res)).encode('utf-8'))
    file_res.write('\n')
    return res

def tokenizer_charNword(txt):
    res = []
    txt = re.sub('\*\*\*', '*', txt)
    for w in list(jieba.cut(txt)):
        if w in STOP_WORDS:
            continue
        res.append(w)
    return res

def build_iters(ftrain='train.tsv',fvalid='valid.tsv',bsz=64, level='char', min_freq=1):

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
    TEXT.build_vocab(train, min_freq = min_freq)
    print('Vocab size: ', len(TEXT.vocab.itos))
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

def build_iters_lm(ftrain='train.tsv',fvalid='valid.tsv',bsz=64, level='char', min_freq=1):

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
    TEXT.build_vocab(train, min_freq=min_freq)
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
    TEXT, LABEL, train_iter, valid_iter = build_iters(level='char', min_freq=1)
    file_res.close()
    # for sample in train_iter:
    #     print(sample.seq1)
    # word_dict = sorted(word_dict.items(), key=lambda tuple: tuple[1])
    # print(word_dict)