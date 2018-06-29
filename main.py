#coding:utf-8
import argparse
import onmt.opts as opts
from preproc import iters
from onmt import model_builder
from onmt.encoders import transformer as enc
import torch
# from phrase_sim_4way import PhraseSim
# from phrase_sim_4way import init_model
# from phrase_sim_gru import PhraseSim
# from phrase_sim_gru import Encoder
# from phrase_sim_gru import init_model
from phrase_sim_pooling import PhraseSim
from phrase_sim_pooling import PoolingEncoder
from phrase_sim_pooling import init_model
from torchtext import data
import torchtext
import os
from preproc.iters import PAD_WORD
import codecs
import chardet

def to_test(val_iter, model):
    model.eval()

    index_lst = []
    pred_lst = []
    with torch.no_grad():
        for i, sample in enumerate(val_iter):
            index, seq1, seq2 = sample.index, sample.seq1,\
                              sample.seq2
            # print(index, seq1, seq2)
            # print((i+.0)/len(valid_iter))

            seq1 = seq1.to(device)
            seq2 = seq2.to(device)

            # seq : (seq_len,bsz)
            # lbl : (bsz)
            probs = model(seq1, seq2)
            # probs : (bsz)
            pred = probs.max(dim=1)[1].cpu()
            pred_lst.extend(pred.numpy())
            index_lst.extend(index.numpy())

    return index_lst, pred_lst

def change_file_encoding(f):
    with open(f, 'r') as f_in:
        with open(f+'.re-encoded', 'w') as f_out:
            f_cnt = f_in.read()
            enc_in = chardet.detect(f_cnt)
            f_cnt_renc = f_cnt.decode(enc_in['encoding']).encode('utf-8')
            f_out.write(f_cnt_renc)

    # with codecs.open(f,'r',encoding='utf-8') as f_in:
    #     with codecs.open(f+'.re-encoded','w',encoding='utf-8') as f_out:
    #         f_out.writelines(f_in.readlines())
    return f+'.re-encoded'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.test_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    INDEX = torchtext.data.Field(sequential=False, use_vocab=False)

    TEXT, LABEL, train_iter, valid_iter = \
        iters.build_iters(ftrain=opt.ftrain, fvalid=opt.fvalid, bsz=opt.batch_size, level=opt.level)

    ftest = change_file_encoding(opt.ftest)

    test = data.TabularDataset(path=ftest, format='tsv',
                                fields=[
                                    ('index', INDEX),
                                    ('seq1', TEXT),
                                    ('seq2', TEXT),
                                ])

    test_iter = data.Iterator(test, batch_size=opt.batch_size,
                               sort=False, repeat=False)
    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    encoder = PoolingEncoder(len(TEXT.vocab.stoi),
                      opt.rnn_size,
                      TEXT.vocab.stoi[PAD_WORD],
                      opt.enc_layers,
                      opt.dropout)
    model = PhraseSim(encoder, 100).to(device)
    init_model(opt, model)

    if opt.load_idx != -1:
        basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
        model_fname = basename + ".model"
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_dict = torch.load(model_fname, map_location=location)
        model.load_state_dict(model_dict)
        print("Loading model from '%s'" % (model_fname))

    indices, pred = to_test(test_iter,model)
    with open(opt.fout, 'w') as f:
        for index, y in zip(indices, pred):
            f.write(str(index) + '\t%d\n' % int(y))
