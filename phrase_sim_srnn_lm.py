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
import sys
import crash_on_ipy
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)

NACT=3
NONLINEAR=nn.Tanh
PUSH=0
POP=1
NOOP=2
USE_STACK=True

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

    def __init__(self, encoder, dropout):
        super(PhraseSim, self).__init__()
        self.encoder = encoder
        self.avg = Avg()
        self.MultiWay =MultiWay()
        self.generator = nn.Sequential(
            nn.Linear(self.MultiWay.nways * encoder.odim,
                      1 * encoder.odim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1*encoder.odim, 2),
            nn.Softmax())
        self.dropout = nn.Dropout(dropout)

        self.W_dis = nn.\
            Parameter(torch.Tensor(encoder.odim,
                                   encoder.odim))

    def forward(self, seq1, seq2):
        # seq1 = seq1.unsqueeze(2)
        # seq2 = seq2.unsqueeze(2)

        outputs1, hidden1, stacks1 = self.encoder(seq1)
        outputs2, hidden2, stacks2 = self.encoder(seq2)

        cat_res = self.MultiWay(hidden1, hidden2)
        cat_res = self.dropout(cat_res)

        probs = self.generator(cat_res)
        we_T = self.encoder.embedding.weight.transpose(0, 1)

        logits1 = torch.matmul(outputs1, self.W_dis.matmul(we_T))
        logits2 = torch.matmul(outputs2, self.W_dis.matmul(we_T))

        return probs, logits1, logits2

class Attention(nn.Module):

    def __init__(self, hdim):
        super(Attention, self).__init__()
        self.hdim = hdim
        self.generator = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.Tanh(),
            nn.Linear(hdim, 1),
            # nn.Softmax(dim=0)
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs, mask):
        a_raw = self.generator(inputs)
        a_raw.masked_fill_(mask.unsqueeze(-1), -float('inf'))
        a = self.softmax(a_raw)
        return (inputs * a).sum(dim=0)

class Encoder(nn.Module):

    def __init__(self, voc_size, hdim, padding_idx,
                 n_layers=1, dropout=0.5, bidirection=False):
        super(Encoder, self).__init__()
        self.hdim = hdim
        self.embedding = nn.Embedding(voc_size,
                                      hdim,
                                      padding_idx=padding_idx)
        self.padding_idx = padding_idx
        self.n_layers = n_layers
        self.bidirection = bidirection
        self.gru = nn.GRU(hdim, hdim, n_layers,
                          dropout, bidirectional=bidirection)
        self.odim = hdim * 2 if bidirection else hdim
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(self.odim)

    def forward(self, inputs, hidden=None):
        embs = self.embedding(inputs)
        mask = inputs.data.eq(self.padding_idx)
        mask_embs = mask.unsqueeze(-1).expand_as(embs)
        embs.masked_fill_(mask_embs, 0)

        embs = self.dropout(embs)

        outputs, hidden = self.gru(embs, hidden)
        final_hidden = self.attention(outputs, mask)

        return outputs, final_hidden

def shift_matrix(n):
    W_up=np.eye(n)
    for i in range(n-1):
        W_up[i,:]=W_up[i+1,:]
    W_up[n-1,:]*=0
    W_down=np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:]=W_down[i-1,:]
    W_down[0,:]*=0
    return W_up,W_down

class EncoderSRNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 padding_idx, nstack,
                 stack_depth, stack_size,
                 stack_elem_size):
        super(EncoderSRNN, self).__init__()
        # here input dimention is equal to hidden dimention
        self.hidden_size = hidden_size
        self.nstack=nstack
        self.stack_size=stack_size
        self.stack_depth=stack_depth
        self.stack_elem_size=stack_elem_size
        self.embedding = nn.Embedding(input_size,
                                      hidden_size,
                                      padding_idx=padding_idx)
        self.padding_idx = padding_idx
        # self.embedding = nn.Embedding(input_size,
        #                               hidden_size)
        #
        self.nonLinear=NONLINEAR()
        self.hid2hid=nn.Linear(hidden_size,hidden_size)
        self.input2hid=nn.Linear(hidden_size,hidden_size)
        self.hid2act=nn.Linear(hidden_size,nstack*NACT)
        self.hid2stack=nn.Linear(hidden_size,nstack*stack_elem_size)
        self.read_stack=nn.Linear(stack_elem_size*stack_depth,hidden_size)

        self.gru = nn.GRUCell(hidden_size, hidden_size)

        self.empty_elem = \
            nn.Parameter(torch.randn(1, self.stack_elem_size))

        self.init_hidden = nn.Parameter(torch.zeros(1, self.hidden_size),
                                        requires_grad=False)

        W_up, W_down = shift_matrix(stack_size)
        self.W_up = nn.Parameter(torch.Tensor(W_up))
        self.W_down = nn.Parameter(torch.Tensor(W_down))

        self.odim = hidden_size
        self.attention = Attention(self.odim)

    def update_stack(self, stacks,
                     p_push, p_pop, p_noop, push_vals):
        # stacks: bsz * nstack * stacksz * stackelemsz
        # p_push, p_pop, p_noop: bsz * nstack * 1
        # push_vals: bsz * nstack * stack_elem_size
        # p_xact: bsz * nstack

        # p_push, p_pop, p_noop: bsz * nstack * 1 * 1
        batch_size = stacks.shape[0]
        p_push = p_push.unsqueeze(3)
        p_pop = p_pop.unsqueeze(3)
        p_noop = p_noop.unsqueeze(3)

        # stacks: bsz * nstack * stacksz * stackelemsz
        stacks = p_push * (self.W_down.matmul(stacks))+\
                 p_pop * (self.W_up.matmul(stacks))+ \
                 p_noop * stacks

        # p_push: bsz * nstack * 1
        p_push=p_push.squeeze(3)
        stacks[:,:,0,:]=(p_push * push_vals)

        stacks[:,:,self.stack_size-1,:]=\
            self.empty_elem.expand(batch_size,self.nstack,self.stack_elem_size)

        return stacks

    def forward(self, inputs):

        bsz = inputs.shape[1]
        hidden = self.init_hidden(bsz)
        stacks = self.init_stack(bsz)

        # inputs: length * bsz
        # stacks: bsz * nstack * stacksz * stackelemsz
        embs = self.embedding(inputs)
        # inputs(length,bsz)->embd(length,bsz,embdsz)
        mask = inputs.data.eq(self.padding_idx)
        mask_embs = mask.unsqueeze(-1).expand_as(embs)
        embs.masked_fill_(mask_embs, 0)

        embs = self.dropout(embs)
        # batch_size=inputs.shape[1]

        outputs=[]
        for input in embs:
            # input: bsz * embdsz
            # hidden: bsz * hidden_size
            cur_hidden=self.gru(input,hidden)


            if USE_STACK:
                # # stack_vals: bsz * nstack * (stack_depth * stack_elem_size)
                act = self.hid2act(hidden)
                act = act.view(-1, self.nstack, NACT)
                # act: bsz * nstack * 3
                act = F.softmax(act, dim=2)
                # p_push, p_pop, p_noop: bsz * nstack * 1
                p_push, p_pop, p_noop = act.chunk(NACT, dim=2)

                # push_vals: bsz * (nstack * stack_elem_size)
                push_vals = self.hid2stack(hidden)
                push_vals = push_vals.view(-1, self.nstack, self.stack_elem_size)
                # push_vals: bsz * nstack * stack_elem_size
                push_vals = self.nonLinear(push_vals)
                stacks = self.update_stack(stacks, p_push, p_pop, p_noop, push_vals)

            hidden=cur_hidden
            output = hidden
            outputs.append(output)

        outputs = torch.stack(outputs)
        final_hidden = self.attention(outputs, mask)

        return outputs, final_hidden, stacks

    def init_stack(self, batch_size):
        return self.empty_elem.expand(batch_size,
                                      self.nstack,
                                      self.stack_size,
                                      self.stack_elem_size).\
                                      contiguous()

    def init_hidden(self, batch_size):
        return self.hidden_size.expand(batch_size, self.hidden_size)

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
        'loss': losses['loss'],
        'loss_ps': losses['loss_ps'],
        'loss_lm': losses['loss_lm'],
        'accuracy': accurs,
        'precs': precs,
        'recalls': recalls,
        'f1s': f1s
    }
    open(train_fname, 'wt').write(json.dumps(content))

def train_batch(sample, model, criterion, optim, lm_coef):
    model.train()

    model.zero_grad()
    seq1, seq2, lbl = sample.seq1, \
                      sample.seq2, \
                      sample.lbl

    seq1 = seq1.to(device)
    seq2 = seq2.to(device)
    tar1 = seq1[1:]
    tar2 = seq2[1:]
    lbl = lbl.to(device)
    # seq : (seq_len,bsz)
    # lbl : (bsz)
    probs, logits1, logits2 = model(seq1[:-1], seq2[:-1])
    voc_size = model.encoder.embedding.num_embeddings
    # decoder_outputs : (1,bsz,hdim)
    # probs : (bsz)

    loss_ps = criterion['ps'](probs, lbl)
    loss_lm1 = criterion['lm'](logits1.view(-1, voc_size), tar1.view(-1))
    loss_lm2 = criterion['lm'](logits2.view(-1, voc_size), tar2.view(-1))
    loss = (loss_ps + lm_coef * (loss_lm1+loss_lm2)/2)/(1+lm_coef)
    loss.backward()
    optim.step()

    return {'loss':loss.data.item(),
            'ps':loss_ps.data.item(),
            'lm':((loss_lm1+loss_lm2)/2).data.item()}

def restore_log(opt):
    basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
    json_fname = basename + ".json"
    history = json.loads(open(json_fname, "rt").read())

    return history['loss'],history['accuracy'],\
           history['precs'],history['recalls'],history['f1s']

def train(train_iter, val_iter, epoch, model,
          optim, criterion, opt):

    losses=[]
    losses_ps = []
    losses_lm = []
    accurs=[]
    f1s=[]
    precs=[]
    recalls=[]
    losses_log=[]
    losses_ps_log=[]
    losses_lm_log = []

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

            loss = train_batch(sample,model,criterion,optim,opt.lm_coef)
            losses.append(loss['loss'])
            losses_ps.append(loss['ps'])
            losses_lm.append(loss['lm'])
            percent = (i+.0)/len(train_iter)
            progress_bar(percent,loss['loss'],epoch)

        accurracy, precision, recall, f1 = valid(val_iter,model)
        loss_mean = np.array(losses[-nbatch:]).mean()
        loss_ps_mean = np.array(losses_ps[-nbatch:]).mean()
        loss_lm_mean = np.array(losses_lm[-nbatch:]).mean()
        print("Valid: accuracy:%.3f precision:%.3f recall:%.3f f1:%.3f avg_loss(total/ps/lm):%.4f/%.4f/%.4f" %
              (accurracy, precision, recall, f1, loss_mean, loss_ps_mean, loss_lm_mean))
        accurs.append(accurracy)
        precs.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        losses_log.append(loss_mean)
        losses_ps_log.append(loss_ps_mean)
        losses_lm_log.append(loss_lm_mean)

        if (epoch+1) % save_per == 0:
            save_checkpoint(model,epoch,{'loss':losses_log,
                                         'loss_ps':losses_ps_log,
                                         'loss_lm':losses_lm_log},
                            accurs,precs,recalls,f1s,opt.exp)

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
            probs, _, _ = model(seq1, seq2)
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
        iters.build_iters_lm(ftrain=opt.ftrain, fvalid=opt.fvalid,
                             bsz=opt.batch_size, level=opt.level,
                             min_freq=opt.min_freq)

    class_probs = dataset_bias(train_iter)
    print('Class probs: ', class_probs)
    # cweights = class_weight(class_probs, opt.label_smoothing)
    cweights = {
        'wneg': 1 - opt.pos_weight,
        'wpos': opt.pos_weight
    }
    print('Class weights: ', cweights)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)
    # encoder = Encoder(len(TEXT.vocab.stoi),
    #                   opt.rnn_size,
    #                   TEXT.vocab.stoi[PAD_WORD],
    #                   opt.enc_layers,
    #                   opt.dropout,
    #                   opt.bidirection)

    encoder = EncoderSRNN(len(TEXT.vocab.stoi),
                          opt.rnn_size,
                          TEXT.vocab.stoi[PAD_WORD],
                          opt.nstack,
                          opt.stack_depth,
                          opt.stack_size,
                          opt.stack_elem_size)

    model = PhraseSim(encoder, opt.dropout).to(device)
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
    criterion_ps = nn.CrossEntropyLoss(weight=weight)
    criterion_lm = nn.CrossEntropyLoss(ignore_index=TEXT.vocab.stoi[iters.PAD_WORD])
    criterion = {'ps': criterion_ps,
                 'lm': criterion_lm}
    epoch = {'start': opt.load_idx + 1 if opt.load_idx != -1 else 0,
             'end': opt.nepoch,
             'save_per': opt.save_per}

    train(train_iter, valid_iter, epoch,
          model, optim, criterion, opt)