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
SAVE_PER = 10

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

class FourWay(nn.Module):

    def forward(self, u1, u2):
        way1 = u1
        way2 = u2
        way3 = torch.abs(u1-u2)
        way4 = u1*u2

        return torch.cat([way1,way2,way3,way4],dim=1)


class PhraseSim(nn.Module):

    def __init__(self, encoder, device):
        super(PhraseSim, self).__init__()
        self.encoder = encoder
        self.avg = Avg()
        self.fourway =FourWay()
        self.generator = nn.Sequential(
            nn.Linear(4*encoder.odim,1*encoder.odim),
            nn.ReLU(),
            nn.Linear(1*encoder.odim,1),
            nn.Sigmoid())
        self.device = device

    def forward(self, seq1, seq2):
        # seq1 = seq1.unsqueeze(2)
        # seq2 = seq2.unsqueeze(2)

        bsz = seq1.shape[1]
        hidden = \
            self.encoder.\
                init_hidden(bsz, self.device)
        stacks = \
            self.encoder.\
                init_stack(bsz, self.device)

        _, hidden1, stacks = self.encoder(seq1, hidden, stacks)
        _, hidden2, stacks = self.encoder(seq2, hidden, stacks)

        cat_res = self.fourway(hidden1,hidden2)

        probs = self.generator(cat_res)

        return probs

def create_stack(stack_size,stack_elem_size):
    return np.array([([EMPTY_VAL] * stack_elem_size)] * stack_size)

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
                 nstack, stack_depth, stack_size,
                 stack_elem_size):
        super(EncoderSRNN, self).__init__()
        # here input dimention is equal to hidden dimention
        self.hidden_size = hidden_size
        self.nstack=nstack
        self.stack_size=stack_size
        self.stack_depth=stack_depth
        self.stack_elem_size=stack_elem_size
        # self.embedding = nn.Embedding(input_size,
        #                               hidden_size,
        #                               padding_idx=PAD)
        self.embedding = nn.Embedding(input_size,
                                      hidden_size)

        self.nonLinear=NONLINEAR()
        self.hid2hid=nn.Linear(hidden_size,hidden_size)
        self.input2hid=nn.Linear(hidden_size,hidden_size)
        self.hid2act=nn.Linear(hidden_size,nstack*NACT)
        self.hid2stack=nn.Linear(hidden_size,nstack*stack_elem_size)
        self.read_stack=nn.Linear(stack_elem_size*stack_depth,hidden_size)

        self.gru = nn.GRUCell(hidden_size, hidden_size)

        self.empty_elem =torch.randn(1,self.stack_elem_size,requires_grad=True)

        W_up, W_down = shift_matrix(stack_size)
        self.W_up = nn.Parameter(torch.Tensor(W_up))
        self.W_down = nn.Parameter(torch.Tensor(W_down))

        self.odim = hidden_size

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

    def forward(self, inputs, hidden, stacks):
        # inputs: length * bsz
        # stacks: bsz * nstack * stacksz * stackelemsz
        embs = self.embedding(inputs)
        # inputs(length,bsz)->embd(length,bsz,embdsz)

        # batch_size=inputs.shape[1]

        outputs=[]
        for input in embs:
            # input: bsz * embdsz
            # input=self.input2hid(input)

            # hidden: bsz * hidden_size
            cur_hidden=self.gru(input,hidden)


            if USE_STACK:
                # # stack_vals: bsz * nstack * (stack_depth * stack_elem_size)
                # stack_vals = stacks[:, :, :self.stack_depth, :].contiguous(). \
                #     view(batch_size,
                #          self.nstack,
                #          self.stack_depth * self.stack_elem_size)
                #
                # # read_res: bsz * nstack * hidden_size
                # read_res = self.read_stack(stack_vals)
                # # read_res: bsz * (nstack * hidden_size)
                # read_res = read_res.view(batch_size, -1)

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

        return outputs, hidden, stacks

    def init_stack(self, batch_size, device):
        return self.empty_elem.expand(batch_size,
                                      self.nstack,
                                      self.stack_size,
                                      self.stack_elem_size).\
                                      contiguous().to(device)
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size,self.hidden_size).to(device)

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
    lbl = lbl.type(torch.FloatTensor)

    bs_weight = lbl.clone().\
        apply_(lambda x:class_weight['wneg'] if x == 0 else class_weight['wpos'])

    criterion.weight = bs_weight.to(device)

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

    if opt.load_idx != -1:
        losses,accurs,\
        precs,recalls,\
        f1s=restore_log(opt)

    epoch_start = epoch['start']
    epoch_end = epoch['end']

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

        if (epoch+1) % SAVE_PER == 0:
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

    TEXT, LALEBL, train_iter, valid_iter = \
        iters.build_iters(ftrain='train.tsv', bsz=opt.batch_size)

    class_probs = dataset_bias(train_iter)
    print('Class probs: ', class_probs)
    cweights = class_weight(class_probs, opt.label_smoothing)
    print('Class weights: ', cweights)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)
    encoder = EncoderSRNN(len(TEXT.vocab.stoi),
                          opt.rnn_size,
                          opt.nstack,
                          opt.stack_depth,
                          opt.stack_size,
                          opt.stack_elem_size)

    model = PhraseSim(encoder, device).to(device)
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

    train(train_iter, valid_iter, epoch,
          model, optim, criterion, opt, cweights)