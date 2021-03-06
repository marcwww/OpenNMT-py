from __future__ import print_function
import argparse
import onmt.opts as opts
from preproc import iters
from torch import nn
import torch
import json
from preproc.iters import PAD_WORD
from torch.nn.init import xavier_uniform_
import sklearn
from sklearn import metrics
import logging
import numpy as np
from onmt.utils import optimizers
import crash_on_ipy

LOGGER = logging.getLogger(__name__)
SAVE_PER = 5

class Avg(nn.Module):

    def forward(self, mem_bank):
        # mem_bank: (seq_len, bsz, hdim)
        len, bsz, hdim = mem_bank.shape

        return mem_bank.sum(dim=0)/len

class FourWay(nn.Module):

    def forward(self, u1, u2):
        way1 = u1
        way2 = u2
        way3 = torch.abs(u1-u2)
        way4 = u1*u2

        return torch.cat([way1,way2,way3,way4],dim=1)


class PhraseSim(nn.Module):

    def __init__(self, encoder_ch, encoder_w, clf_dim, dropout):
        super(PhraseSim, self).__init__()
        self.encoder_ch = encoder_ch
        self.encoder_w = encoder_w
        self.avg = Avg()
        self.fourway =FourWay()
        cat_dim = encoder_ch.odim + encoder_w.odim
        self.generator = nn.Sequential(
            nn.Linear(4*cat_dim,
                      clf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_dim,1),
            nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1ch, seq2ch,
                seq1w, seq2w):

        _, hidden1ch = self.encoder_ch(seq1ch)
        _, hidden2ch = self.encoder_ch(seq2ch)

        cat_res_ch = self.fourway(hidden1ch,hidden2ch)
        cat_res_ch = self.dropout(cat_res_ch)

        _, hidden1w = self.encoder_w(seq1w)
        _, hidden2w = self.encoder_w(seq2w)

        cat_res_w = self.fourway(hidden1w, hidden2w)
        cat_res_w = self.dropout(cat_res_w)

        cat_res = torch.cat([cat_res_ch, cat_res_w], dim=-1)

        probs = self.generator(cat_res)

        return probs

class Encoder(nn.Module):

    def __init__(self, voc_size, hdim, padding_idx,
                 n_layers=1, dropout=0.5, bidirection=True):
        super(Encoder, self).__init__()
        self.hdim = hdim
        self.embedding = nn.Embedding(voc_size,
                                      hdim,
                                      padding_idx=padding_idx)
        self.n_layers = n_layers
        self.gru = nn.GRU(hdim, hdim, n_layers,
                          dropout, bidirectional=bidirection)
        self.odim = hdim * 2 if bidirection else hdim
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden=None):
        embs = self.embedding(inputs)
        embs = self.dropout(embs)

        outputs, hidden = self.gru(embs, hidden)
        # outputs = outputs[:,:,:self.hdim]\
        #           + outputs[:,:,self.hdim:]
        # hidden = hidden[:self.n_layers]
        # hidden = torch.cat([hidden[i] for i in range(self.n_layers)], dim=1)
        final_hidden = outputs[-1]

        return outputs, final_hidden

def progress_bar(percent, last_loss, epoch):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f}/epoch {:d} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), percent, epoch, last_loss), end='')

def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='')

def save_checkpoint(model, epoch,
                    losses, accurs,
                    precs, recalls,
                    f1s, name):
    progress_clean()

    basename = "{}-epoch-{}".format(name,epoch)
    model_fname = basename + ".model"
    # LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    print("Saving model checkpoint to: '%s'" % (model_fname))
    torch.save(model.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    # LOGGER.info("Saving model training history to '%s'", train_fname)
    print("Saving model training history to '%s'" % (train_fname))
    content = {
        'loss': losses,
        'accuracy': accurs,
        'precs': precs,
        'recalls': recalls,
        'f1s': f1s
    }
    open(train_fname, 'wt').write(json.dumps(content))

def train_batch(sample, model, criterion, optim, class_weight):
    model.train()

    sample_ch, sample_w = sample

    model.zero_grad()
    seq1ch, seq2ch, lbl = sample_ch.seq1, \
                      sample_ch.seq2, \
                      sample_ch.lbl

    seq1w, seq2w, lbl = sample_w.seq1, \
                          sample_w.seq2, \
                          sample_w.lbl

    seq1ch = seq1ch.to(device)
    seq2ch = seq2ch.to(device)
    seq1w = seq1w.to(device)
    seq2w = seq2w.to(device)

    lbl = lbl.type(torch.FloatTensor)

    bs_weight = lbl.clone().\
        apply_(lambda x:class_weight['wneg'] if x == 0 else class_weight['wpos'])

    criterion.weight = bs_weight.to(device)

    lbl = lbl.to(device)
    # seq : (seq_len,bsz)
    # lbl : (bsz)
    probs = model(seq1ch, seq2ch, seq1w, seq2w)
    # decoder_outputs : (1,bsz,hdim)
    # probs : (bsz)

    loss = criterion(probs, lbl)
    loss.backward()
    # print(sum-param_sum(model.parameters()),sum,param_sum(model.parameters()))
    # sum=param_sum(model.parameters())
    # clip_grads(model)
    optim.step()

    return loss

def restore_log(opt):
    basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
    json_fname = basename + ".json"
    history = json.loads(open(json_fname, "rt").read())

    return history['loss'],history['accuracy'],\
           history['precs'],history['recalls'],history['f1s']

def train(train_iter_ch, val_iter_ch,
          train_iter_w, val_iter_w,
          epoch, model,
          optim, criterion, opt, class_weight):
    # sum=param_sum(model.parameters())
    losses=[]
    accurs=[]
    f1s=[]
    precs=[]
    recalls=[]

    if opt.load_idx != -1:
        losses,accurs,\
        precs,recalls,\
        f1s=restore_log(opt)

    epoch_start = epoch['start']
    epoch_end = epoch['end']

    # valid(val_iter,model)

    for epoch in range(epoch_start,epoch_end):
        nbatch = 0
        for i, sample in enumerate(zip(train_iter_ch, train_iter_w)):
            nbatch += 1

            loss = train_batch(sample,model,criterion,optim,class_weight)

            loss_val = loss.data.item()
            losses.append(loss_val)
            percent = (i+.0)/len(train_iter_ch)
            progress_bar(percent,loss_val,epoch)

        accurracy, precision, recall, f1 = \
            valid(val_iter_ch, valid_iter_w,model)
        print("Valid: accuracy:%.3f precision:%.3f recall:%.3f f1:%.3f avg_loss:%.4f" %
              (accurracy, precision, recall, f1, np.array(losses[-nbatch:]).mean()))
        accurs.extend([accurracy for _ in range(nbatch)])
        precs.extend([precision for _ in range(nbatch)])
        recalls.extend([recall for _ in range(nbatch)])
        f1s.append([f1 for _ in range(nbatch)])

        if (epoch+1) % SAVE_PER == 0:
            save_checkpoint(model,epoch,losses,accurs,precs,recalls,f1s,opt.exp)

def valid(val_iter_ch, val_iter_w, model):
    model.eval()

    pred_lst = []
    lbl_lst = []
    with torch.no_grad():
        for i, sample in enumerate(zip(val_iter_ch, val_iter_w)):

            sample_ch,sample_w = sample
            seq1ch, seq2ch, lbl = sample_ch.seq1,\
                              sample_ch.seq2,\
                              sample_ch.lbl

            seq1w, seq2w, lbl = sample_w.seq1, \
                                  sample_w.seq2, \
                                  sample_w.lbl

            seq1ch = seq1ch.to(device)
            seq2ch = seq2ch.to(device)
            seq1w = seq1w.to(device)
            seq2w = seq2w.to(device)
            lbl = lbl.type(torch.FloatTensor)

            # seq : (seq_len,bsz)
            # lbl : (bsz)
            probs = model(seq1ch, seq2ch, seq1w, seq2w)
            # probs : (bsz)
            pred = probs.cpu().apply_(lambda x: 0 if x < 0.5 else 1)
            pred_lst.extend(pred.numpy())
            lbl_lst.extend(lbl.numpy())

            # print((i+0.0)/len(val_iter))

    accurracy = metrics.accuracy_score(np.array(lbl_lst),np.array(pred_lst))
    precision = metrics.precision_score(np.array(lbl_lst),np.array(pred_lst))
    recall =metrics.recall_score(np.array(lbl_lst),np.array(pred_lst))
    f1 = metrics.f1_score(np.array(lbl_lst),np.array(pred_lst))

    return accurracy, precision, recall, f1

def init_model(model_opt, model):

    def param_sum(param):
        res = 0
        for p in param:
            res += p.data.cpu().numpy().sum()

        return res

    print('Param sum before init: ', param_sum(model.parameters()))

    if model_opt.param_init != 0.0:
        print('Intializing model parameters.')
        for p in model.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)

    if model_opt.param_init_glorot:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    print('Param sum after init: ', param_sum(model.parameters()))

def dataset_bias(train_iter):
    npos = 0
    nneg = 0
    for sample in train_iter:
        b_npos = sample.lbl.numpy().sum()
        npos += b_npos
        nneg += sample.lbl.shape[0] - b_npos

    return {'ppos':npos/(npos+nneg+.0),
            'pneg':nneg/(npos+nneg+.0)}

def class_weight(class_probs, e):
    ppos = class_probs['ppos']
    pneg = class_probs['pneg']

    return {'wpos':(1-e)*0.5+e*(1-ppos),
            'wneg':(1-e)*0.5+e*(1-pneg)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()

    TEXT_ch, LALEBL, train_iter_ch, valid_iter_ch = \
        iters.build_iters(ftrain=opt.ftrain, fvalid=opt.fvalid,
                          bsz=opt.batch_size, level='char')

    class_probs = dataset_bias(train_iter_ch)
    print('Class probs: ', class_probs)
    cweights = class_weight(class_probs, opt.label_smoothing)
    print('Class weights: ', cweights)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)
    encoder_ch = Encoder(len(TEXT_ch.vocab.stoi),
                         opt.rnn_size,
                         TEXT_ch.vocab.stoi[PAD_WORD],
                         opt.enc_layers,
                         opt.dropout,
                         opt.bidirection)

    TEXT_w, LALEBL, train_iter_w, valid_iter_w = \
        iters.build_iters(ftrain=opt.ftrain, fvalid=opt.fvalid,
                          bsz=opt.batch_size, level='word')

    encoder_w = Encoder(len(TEXT_w.vocab.stoi),
                         opt.rnn_size,
                         TEXT_w.vocab.stoi[PAD_WORD],
                         opt.enc_layers,
                         opt.dropout,
                         opt.bidirection)

    model = PhraseSim(encoder_ch, encoder_w, opt.clf_dim, opt.dropout).to(device)
    init_model(opt, model)

    # print(model.state_dict())
    if opt.load_idx != -1:
        basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
        model_fname = basename + ".model"
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_dict = torch.load(model_fname, map_location=location)
        model.load_state_dict(model_dict)
        print("Loading model from '%s'" % (model_fname))

    # model.generator = generator.to(device)
    optim = optimizers.build_optim(model, opt, None)
    criterion = nn.BCELoss(size_average=True)
    epoch = {'start': opt.load_idx if opt.load_idx != -1 else 0,
             'end': 10000}

    train(train_iter_ch, valid_iter_ch,
          train_iter_w, valid_iter_w,
          epoch,
          model, optim, criterion, opt, cweights)