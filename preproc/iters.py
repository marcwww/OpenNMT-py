import torchtext
import os
import jieba

from torchtext import data

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'

# HOME=os.path.abspath('.')
HOME=os.path.abspath('..')
DATA=os.path.join(HOME,'data_folder')

jieba.load_userdict(os.path.join(DATA,'dict.txt'))

def tokenizer_word(txt):
    return list(jieba.cut(txt))

def tokenizer_char(txt):
    return list(txt)

def build_iters(ftrain='train.tsv',fvalid='valid.tsv',bsz=64, level='char'):
    TEXT = torchtext.data.Field(sequential=True,
                                tokenize=tokenizer_word if level == 'word' else tokenizer_char,
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

if __name__ == '__main__':
    TEXT, LABEL, train_iter, valid_iter = build_iters()
    for sample in train_iter:
        print(sample.seq1)
