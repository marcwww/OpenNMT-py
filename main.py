import argparse
import onmt.opts as opts
from preproc import iters
from onmt import model_builder
from onmt.encoders import transformer as enc
import torch
from phrase_sim_4way import PhraseSim
from phrase_sim_4way import init_model
from torchtext import data
import torchtext
import os

def to_valid(val_iter, model):
    model.eval()

    index_lst = []
    pred_lst = []
    with torch.no_grad():
        for i, sample in enumerate(val_iter):
            index, seq1, seq2 = sample.index, sample.seq1,\
                              sample.seq2\

            seq1 = seq1.to(device)
            seq2 = seq2.to(device)

            # seq : (seq_len,bsz)
            # lbl : (bsz)
            probs = model(seq1, seq2)
            # probs : (bsz)
            pred = probs.cpu().apply_(lambda x: 0 if x < 0.5 else 1)
            pred_lst.extend(pred.numpy())
            index_lst.extend(index.numpy())

    return index_lst, pred_lst

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.valid_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    INDEX = torchtext.data.Field(sequential=False, use_vocab=False)

    TEXT, LABEL, train_iter, valid_iter = \
        iters.build_iters(ftrain='train.tsv', bsz=opt.batch_size)

    valid = data.TabularDataset(path=opt.fvalid, format='tsv',
                                fields=[
                                    ('index', INDEX),
                                    ('seq1', TEXT),
                                    ('seq2', TEXT),
                                ])

    valid_iter = data.Iterator(valid, batch_size=opt.batch_size,
                               sort=False, repeat=False)

    embeddings_enc = model_builder.build_embeddings(opt, TEXT.vocab, [])
    encoder = enc.TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                     opt.dropout, embeddings_enc)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    model = PhraseSim(encoder, opt).to(device)
    init_model(opt, model)

    if opt.load_idx != -1:
        basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
        model_fname = basename + ".model"
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_dict = torch.load(model_fname, map_location=location)
        model.load_state_dict(model_dict)
        print("Loading model from '%s'" % (model_fname))

    indices, pred = to_valid(valid_iter,model)
    with open(opt.fout, 'w') as f:
        for index, y in zip(indices, pred):
            f.write(str(index) + '\t%d\n' % int(y))
