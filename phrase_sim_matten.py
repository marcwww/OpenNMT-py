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
from torch.nn import functional as F
import crash_on_ipy

LOGGER = logging.getLogger(__name__)

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

    def __init__(self, gru_encoder, pooling_encoder,dropout, clf_dim):
        super(PhraseSim, self).__init__()
        self.gru_encoder = gru_encoder
        self.pooling_encoder = pooling_encoder
        self.avg = Avg()
        self.fourway =FourWay()
        self.generator = nn.Sequential(
            nn.Linear(4*gru_encoder.odim, clf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_dim, 2))
        self.dropout = nn.Dropout(dropout)
        self.atten = Atten()

    def forward(self, seq1, seq2):
        # seq1 = seq1.unsqueeze(2)
        # seq2 = seq2.unsqueeze(2)

        outputs1, _ = self.gru_encoder(seq1)
        outputs2, _ = self.gru_encoder(seq2)


        outputs1f = outputs1[:, :, :self.gru_encoder.odim / 2]
        outputs1b = outputs1[:, :, self.gru_encoder.odim / 2:]

        outputs2f = outputs1[:, :, :self.gru_encoder.odim / 2]
        outputs2b = outputs1[:, :, self.gru_encoder.odim / 2:]

        u1 = self.pooling_encoder(seq1)
        u2 = self.pooling_encoder(seq2)

        h1f = self.atten(outputs1f, u2)
        h1b = self.atten(outputs1b, u2)

        h2f = self.atten(outputs2f, u1)
        h2b = self.atten(outputs2b, u1)

        # hif, hib : (bsz, hdim)
        h1 = torch.cat([h1f, h1b], dim=1)
        h2 = torch.cat([h2f, h2b], dim=1)

        cat_res = self.fourway(h1, h2)
        cat_res = self.dropout(cat_res)

        probs = self.generator(cat_res)

        return probs

class PoolingEncoder(nn.Module):

    def __init__(self, embedding, hdim, n_layers=1, dropout=0.5):
        super(PoolingEncoder, self).__init__()
        self.hdim = hdim
        self.embedding = embedding
        self.linear = nn.Linear(hdim, hdim)
        self.dropout = nn.Dropout(dropout)
        self.odim = hdim

    def forward(self, inputs, hidden=None):

        # embs: (seq_len, bsz, hdim)
        embs = self.embedding(inputs)
        embs_dropout = self.dropout(embs)
        embs_affine = self.linear(embs_dropout)
        h, _ = torch.max(embs_affine, dim=0, keepdim=False)

        return h

class Atten(nn.Module):

    def forward(self, inputs, u_h):
        # inputs : (seq_len, bsz, hdim)
        # u_h : (bsz, hdim)
        a = torch.matmul(inputs.unsqueeze(-2), u_h.unsqueeze(-1))
        # a : (seq_len, bsz, 1, 1)
        a = a.squeeze(-1)
        # a : (seq_len, bsz, 1)
        a = F.softmax(a, 0)
        # a : (seq_len, bsz, 1)


        # outputs_weighted : (seq_len, bsz, hdim)
        outputs_weighted = inputs * a
        # final_hidden : (bsz, hdim)
        final_hidden = outputs_weighted.sum(0)

        return final_hidden

class GruEncoder(nn.Module):

    def __init__(self, embedding, hdim,
                 n_layers=1, dropout=0.5):
        super(GruEncoder, self).__init__()
        self.hdim = hdim
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.gru = nn.GRU(hdim, hdim, n_layers,
                          dropout, bidirectional=True)
        self.odim = hdim * 2
        # self.u_h = nn.Parameter(torch.Tensor(self.odim))
        self.linear = nn.Linear(self.odim, self.odim)

    def forward(self, inputs, hidden=None):
        embs = self.embedding(inputs)
        embs = self.dropout(embs)
        outputs, hidden = self.gru(embs, hidden)
        outputs = self.linear(outputs)

        # outputs : (seq_len, bsz, odim)
        # a : (seq_len, bsz)
        # a = torch.matmul(outputs, self.u_h)
        # a = F.softmax(a,0)
        # outputs_weighted = outputs * a.unsqueeze(-1)
        # final_hidden = outputs_weighted.sum(0)

        return outputs, hidden

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

    model.zero_grad()
    seq1, seq2, lbl = sample.seq1, \
                      sample.seq2, \
                      sample.lbl

    seq1 = seq1.to(device)
    seq2 = seq2.to(device)
    lbl = lbl.to(device)
    probs = model(seq1, seq2)
    loss = criterion(probs, lbl)
    loss.backward()
    optim.step()

    return loss

def restore_log(opt):
    basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
    json_fname = basename + ".json"
    history = json.loads(open(json_fname, "rt").read())

    return history['loss'],history['accuracy'],\
           history['precs'],history['recalls'],history['f1s']

def train(train_iter, val_iter, epoch, model,
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
    save_per = epoch['save_per']

    # valid(val_iter,model)

    for epoch in range(epoch_start,epoch_end):
        nbatch = 0
        for i, sample in enumerate(train_iter):
            nbatch += 1

            loss = train_batch(sample,model,criterion,optim,class_weight)

            loss_val = loss.data.item()
            losses.append(loss_val)
            percent = (i+.0)/len(train_iter)
            progress_bar(percent,loss_val,epoch)

        accurracy, precision, recall, f1 = valid(val_iter,model)
        print("Valid: accuracy:%.3f precision:%.3f recall:%.3f f1:%.3f avg_loss:%.4f" %
              (accurracy, precision, recall, f1, np.array(losses[-nbatch:]).mean()))
        accurs.extend([accurracy for _ in range(nbatch)])
        precs.extend([precision for _ in range(nbatch)])
        recalls.extend([recall for _ in range(nbatch)])
        f1s.append([f1 for _ in range(nbatch)])

        if (epoch+1) % save_per == 0:
            save_checkpoint(model,epoch,losses,accurs,precs,recalls,f1s,opt.exp)

def valid(val_iter, model):
    model.eval()

    pred_lst = []
    lbl_lst = []
    with torch.no_grad():
        for i, sample in enumerate(val_iter):
            seq1, seq2, lbl = sample.seq1,\
                              sample.seq2,\
                              sample.lbl

            seq1 = seq1.to(device)
            seq2 = seq2.to(device)
            lbl = lbl.type(torch.FloatTensor)

            # seq : (seq_len,bsz)
            # lbl : (bsz)
            probs = model(seq1, seq2)
            # probs : (bsz)
            pred = probs.max(dim=1)[1].cpu()
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

    TEXT, LALEBL, train_iter, valid_iter = \
        iters.build_iters(ftrain=opt.ftrain, fvalid=opt.fvalid,
                          bsz=opt.batch_size, level=opt.level)

    class_probs = dataset_bias(train_iter)
    print('Class probs: ', class_probs)
    cweights = class_weight(class_probs, opt.label_smoothing)
    print('Class weights: ', cweights)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    embedding = nn.Embedding(len(TEXT.vocab.stoi),
                 opt.rnn_size,
                 padding_idx=TEXT.vocab.stoi[PAD_WORD])

    gru_encoder = GruEncoder(embedding,
                             opt.rnn_size,
                             opt.enc_layers,
                             opt.dropout)
    pooling_encoder = \
        PoolingEncoder(embedding,
                       opt.rnn_size,
                       opt.enc_layers,
                       opt.dropout)
    model = PhraseSim(gru_encoder, pooling_encoder, opt.dropout, opt.clf_dim).to(device)
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
    weight = torch.Tensor([cweights['wneg'], cweights['wpos']]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    epoch = {'start': opt.load_idx if opt.load_idx != -1 else 0,
             'end': opt.nepoch,
             'save_per':opt.save_per}

    train(train_iter, valid_iter, epoch,
          model, optim, criterion, opt, cweights)