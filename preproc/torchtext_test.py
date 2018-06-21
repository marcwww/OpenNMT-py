import jieba
from torchtext import data
from torchtext import vocab
from torchtext import datasets
import os
import torch

DATA = os.path.abspath('../data')

def tokenizer(txt):
    return list(jieba.cut(txt))

TEXT = data.Field(sequential=True, tokenize=tokenizer, pad_token='<pad>')
LABEL = data.Field(sequential=False, use_vocab=False)

ftrain = 'train3.tsv'
train = data.TabularDataset(path=os.path.join(DATA,ftrain),format='tsv',
                    fields=[
                        ('seq1',TEXT),
                        ('seq2',TEXT),
                        ('lbl',LABEL)
                    ])
TEXT.build_vocab(train)
train_iter = data.Iterator(train,batch_size=4,sort=False,repeat=False)
# vocab = TEXT.vocab

embedding = torch.nn.Embedding(num_embeddings = len(TEXT.vocab.itos),
                          embedding_dim=10,
                          padding_idx=TEXT.vocab.stoi[TEXT.pad_token]
                          )

for sample in train_iter:
    seq1,seq2,lbl = [getattr(sample, name)
                     for name in ['seq1','seq2','lbl']]
    embedding(seq1.unsqueeze(-1))
    embedding(seq2.unsqueeze(-1))

fvalid = 'train2.tsv'
TEXT2 = data.Field(sequential=True, tokenize=tokenizer, pad_token='<pad>')
valid = data.TabularDataset(path=os.path.join(DATA,fvalid),format='tsv',
                            fields=[
                                ('seq1',TEXT2),
                                ('seq2',TEXT2),
                                ('lbl',LABEL)
                            ])
valid_iter = data.Iterator(valid,batch_size=4,sort=False,repeat=False)
TEXT2.build_vocab(valid)
for sample in valid_iter:
    seq1,seq2,lbl = [getattr(sample, name)
                     for name in ['seq1','seq2','lbl']]
    embedding(seq1.unsqueeze(-1))
    embedding(seq2.unsqueeze(-1))




