import spacy
from torchtext import data
from torchtext import vocab
from torchtext import datasets

spacy_en = spacy.load('en')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

train, val, test = data.TabularDataset.splits(
    path='../data/', train='train.tsv',
    validation='val.tsv', test='test.tsv',
    format='tsv', fields=[('Text', TEXT),
                          ('Label', LABEL)]
)


TEXT.build_vocab(train)
print(TEXT.vocab.stoi['<pad>'])

print(TEXT.vocab.itos)

train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), batch_sizes=(2,1,1),
    device=-1, repeat=False
)

for sample in train_iter:
    print(sample.Text, sample.Label)






