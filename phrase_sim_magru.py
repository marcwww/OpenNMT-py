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
import sys

LOGGER = logging.getLogger(__name__)
SAVE_PER = 5

class Avg(nn.Module):

    def forward(self, mem_bank):
        # mem_bank: (seq_len, bsz, hdim)
        len, bsz, hdim = mem_bank.shape

        return mem_bank.sum(dim=0)/len

class MultiWay(nn.Module):

    def __init__(self):
        super(MultiWay, self).__init__()
        self.nways = 3

    def forward(self, u1, u2):
        # way1 = u1
        # way2 = u2
        way3 = torch.abs(u1-u2)
        way4 = u1*u2
        way5 = torch.max(torch.stack([u1*u1,u2*u2]), dim=0)[0]

        # return torch.cat([way1,way2,way3,way4],dim=1)
        return torch.cat([way3,way4,way5],dim=1)


class PhraseSim(nn.Module):

    def __init__(self, encoder, dropout, k, nslices):
        super(PhraseSim, self).__init__()
        self.encoder = encoder
        self.avg = Avg()
        self.MultiWay =MultiWay()
        self.generator = nn.Sequential(
            nn.Linear(k * nslices, k * nslices),
            nn.ReLU(),
            nn.Linear(k * nslices, 2),
            nn.Softmax(dim = -1))
        self.dropout = nn.Dropout(dropout)
        self.k = k
        self.mutual_attention_layers = \
            nn.ModuleList([MutualAttention(encoder.odim, k) for _ in xrange(nslices)])
        self.nslices = nslices

    def forward(self, seq1, seq2):

        outputs1, hidden1, mask1 = self.encoder(seq1)
        outputs2, hidden2, mask2 = self.encoder(seq2)

        res = torch.cat([self.mutual_attention_layers[i](outputs1, outputs2,
                                    mask1, mask2)
               for i in xrange(self.nslices)], dim=-1)
        probs = self.generator(res)

        return probs

class MutualAttention(nn.Module):

    def __init__(self, hdim, k):
        super(MutualAttention, self).__init__()
        self.hdim = hdim
        self.W = nn.Parameter(torch.Tensor(hdim, hdim))
        self.b = nn.Parameter(torch.Tensor(1))
        self.k = k
        self.relu = nn.ReLU()

    def forward(self, inputs1, inputs2, mask1, mask2):

        re_mask_inputs1 = mask1.data.eq(0).unsqueeze(-1).expand_as(inputs1)
        re_mask_inputs2 = mask2.data.eq(0).unsqueeze(-1).expand_as(inputs2)
        re_mask_inputs1 = re_mask_inputs1.transpose(0, 1).float()
        re_mask_inputs2_T = re_mask_inputs2.transpose(0, 1).transpose(1, 2).float()
        mask_sims = torch.matmul(re_mask_inputs1, re_mask_inputs2_T).data.eq(0)
        bsz = mask_sims.shape[0]
        mask_sims = mask_sims.view(bsz, -1)
        num_elems = mask_sims.shape[1]

        # inputs : (seq_len, bsz, odim)
        # H1 : (bsz, seq_len1, odim)
        H1 = inputs1.transpose(0, 1)
        # H2_T : (bsz, odim, seq_len2)
        H2_T = inputs2.transpose(0, 1).transpose(1, 2)
        S = torch.matmul(H1, self.W.unsqueeze(0).matmul(H2_T)) + self.b

        S = self.relu(S)

        S_flatten = S.view(bsz, -1)
        S_flatten.masked_fill_(mask_sims, -float('inf'))

        # S_flatten : (bsz, seq_len1 * seq_len2)
        k_actual = min(self.k, num_elems)
        # q : (bsz, k_actual)
        q, _ = torch.topk(S_flatten, k_actual, dim=-1)
        q_num_finite = (k_actual - q.data.eq(-float('inf')).sum(-1)).long()

        for i in xrange(bsz):
            q[i].masked_fill_(q[i].data.eq(-float('inf')), q[i, q_num_finite[i]-1])

        result = q.data.new(bsz, self.k)
        result[:, :k_actual] = q
        if k_actual < self.k:
            rest = q[:, -1].unsqueeze(-1).expand(bsz, self.k - k_actual)
            result[:, k_actual:] = rest

        # result : (bsz, self.k)
        return result

class Encoder(nn.Module):

    def __init__(self, voc_size, hdim, padding_idx,
                 n_layers=1, dropout=0.5, bidirection=True):
        super(Encoder, self).__init__()
        self.hdim = hdim
        self.embedding = nn.Embedding(voc_size,
                                      hdim,
                                      padding_idx=padding_idx)
        self.padding_idx = padding_idx
        self.n_layers = n_layers
        self.gru = nn.GRU(hdim, hdim, n_layers,
                          dropout, bidirectional=bidirection)
        self.odim = hdim * 2 if bidirection else hdim
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden=None):
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        mask_embs = mask.unsqueeze(-1).expand_as(embs)
        embs.masked_fill_(mask_embs, 0)
        embs = self.dropout(embs)

        outputs, hidden = self.gru(embs, hidden)
        mask_hiddens = mask.unsqueeze(-1).expand_as(outputs)
        outputs = outputs.clone().masked_fill_(mask_hiddens, 0)
        final_hidden = outputs[-1]

        return outputs, final_hidden, mask

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
    # seq : (seq_len,bsz)
    # lbl : (bsz)
    probs = model(seq1, seq2)
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

def train(train_iter, val_iter, epoch, model,
          optim, criterion, opt, class_weight):
    # sum=param_sum(model.parameters())
    losses=[]
    accurs=[]
    f1s=[]
    precs=[]
    recalls=[]
    losses_log = []

    if opt.load_idx != -1:
        losses,accurs,\
        precs,recalls,\
        f1s=restore_log(opt)

    epoch_start = epoch['start']
    epoch_end = epoch['end']
    save_per = epoch['save_per']

    for epoch in range(epoch_start,epoch_end):
        nbatch = 0
        for i, sample in enumerate(train_iter):
            nbatch += 1

            loss = train_batch(sample,model,criterion,optim,
                               class_weight)

            loss_val = loss.data.item()
            losses.append(loss_val)
            percent = (i+.0)/len(train_iter)
            progress_bar(percent,loss_val,epoch)

        accurracy, precision, recall, f1 = valid(val_iter,model)
        loss_mean = np.array(losses[-nbatch:]).mean()
        print("Valid: accuracy:%.3f precision:%.3f recall:%.3f f1:%.3f avg_loss:%.4f" %
              (accurracy, precision, recall, f1, loss_mean))
        accurs.append(accurracy)
        precs.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        losses_log.append(loss_mean)

        if (epoch+1) % save_per == 0:
            save_checkpoint(model,epoch,losses_log,accurs,precs,recalls,f1s,opt.exp)

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
    with open('%s.arg' % opt.exp, 'w') as f:
        f.write(' '.join(sys.argv[1:]))

    TEXT, LALEBL, train_iter, valid_iter = \
        iters.build_iters(ftrain=opt.ftrain, fvalid=opt.fvalid,
                          bsz=opt.batch_size, level=opt.level)

    class_probs = dataset_bias(train_iter)
    print('Class probs: ', class_probs)
    cweights = class_weight(class_probs, opt.label_smoothing)
    print('Class weights: ', cweights)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)
    encoder = Encoder(len(TEXT.vocab.stoi),
                      opt.rnn_size,
                      TEXT.vocab.stoi[PAD_WORD],
                      opt.enc_layers,
                      opt.dropout,
                      opt.bidirection)
    model = PhraseSim(encoder, opt.dropout, k=opt.k, nslices=opt.nslices).to(device)
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
             'save_per': opt.save_per}

    train(train_iter, valid_iter, epoch,
          model, optim, criterion, opt, cweights)