import os

dataset_filename='atec'
vocab_filename="vocab.%s.%s" % (dataset_filename, 'tokens')

def generate_vocab_file(data_dir):
    vocab = set()
    vocab.add('\t')
    with open(os.path.join(data_dir, dataset_filename),
              'r',
              encoding='UTF-8') as f:
        for line in f:
            seq1, seq2, lbl = line.split('\t')
            for w in seq1.split(' '):
                vocab.add(w)

            for w in seq2.split(' '):
                vocab.add(w)

    with open(os.path.join(data_dir, vocab_filename),
              'w',
              encoding='UTF-8') as f:
        for w in vocab:
            f.write(w)
            f.write('\n')

data_dir=os.path.join(os.path.expandvars('$HOME'),'t2t_data')

generate_vocab_file(data_dir)