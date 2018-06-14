import argparse
import onmt.opts as opts
from preproc import iters
from onmt import model_builder
from onmt.utils import optimizers
from itertools import count
from onmt.encoders import transformer as enc
from onmt.decoders import transformer_ps as dec
from torch import nn
import torch
from preproc.iters import BOS_WORD
import crash_on_ipy
import numpy as np
import logging
import json
from torch import optim
from sklearn import metrics
import json
from torch.nn.init import xavier_uniform_

LOGGER = logging.getLogger(__name__)
SAVE_PER = 10

def clip_grads(model):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

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

    def __init__(self, encoder, opt):
        super(PhraseSim, self).__init__()
        self.encoder = encoder
        self.avg = Avg()
        self.fourway =FourWay()
        # self.generator = nn.Sequential(
        #     nn.Linear(4*opt.rnn_size,1*opt.rnn_size),
        #     nn.ReLU(),
        #     nn.Linear(1*opt.rnn_size,1),
        #     nn.Sigmoid())
        self.generator = nn.Sequential(
            nn.Linear(4*opt.rnn_size,1*opt.rnn_size),
            nn.ReLU(),
            nn.Linear(1*opt.rnn_size,2),
            nn.Softmax())

    def forward(self, seq1, seq2):
        seq1 = seq1.unsqueeze(2)
        seq2 = seq2.unsqueeze(2)

        _, memory_bank1 = self.encoder(seq1)
        _, memory_bank2 = self.encoder(seq2)
        # memory_bank: (seq_len, bsz, hdim)

        mem_avg1 = self.avg(memory_bank1)
        mem_avg2 = self.avg(memory_bank2)

        cat_res = self.fourway(mem_avg1,mem_avg2)

        # probs: (bsz, 2)
        probs = self.generator(cat_res)

        return probs

def progress_bar(percent, last_loss, epoch):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f}/epoch {:d} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), percent, epoch, last_loss), end='')

def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')

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

def param_sum(param):
    res=0
    for p in param:
        res+=p.data.cpu().numpy().sum()

    return res

def param_del(param_lst1,param_lst2):
    res=0
    for p1,p2 in zip(param_lst1,param_lst2):
        res+=np.abs((p1-p2).data.cpu().numpy().sum())

    return res

def label_smoothing(lbl, class_probs, e):
    return lbl.clone()\
        .apply_(lambda x:(1-e)*x+e*class_probs['ppos' if x==1 else 'pneg'])

def train_batch(sample, model, criterion, optim, class_probs, opt):
    model.train()

    model.zero_grad()
    seq1, seq2, lbl = sample.seq1, \
                      sample.seq2, \
                      sample.lbl

    seq1 = seq1.to(device)
    seq2 = seq2.to(device)
    lbl = lbl.type(torch.FloatTensor)

    lbl = label_smoothing(lbl, class_probs, opt.label_smoothing)

    # bs_weight = lbl.clone().\
    #     apply_(lambda x:class_weight['wneg'] if x == 0 else class_weight['wpos'])
    #
    # criterion.weight = bs_weight.to(device)
    lbl = lbl.to(device)
    # seq : (seq_len,bsz)
    # lbl : (bsz)
    probs = model(seq1, seq2)
    # decoder_outputs : (1,bsz,hdim)
    # probs : (bsz, 2)

    loss = criterion(probs, lbl)
    loss.backward()
    # print(sum-param_sum(model.parameters()),sum,param_sum(model.parameters()))
    # sum=param_sum(model.parameters())
    # clip_grads(model)
    optim.step()

    return loss.data.item()

def train_batches(samples, model, criterion, optim, class_probs, opt):
    model.train()

    model.zero_grad()
    losses=[]
    for sample in samples:
        seq1, seq2, lbl = sample.seq1, \
                          sample.seq2, \
                          sample.lbl

        seq1 = seq1.to(device)
        seq2 = seq2.to(device)
        lbl = lbl.type(torch.FloatTensor)

        lbl_smoothed = label_smoothing(lbl, class_probs, opt.label_smoothing).unsqueeze(1)
        # lbl_smoothed : (bsz, 1)

        lbl = lbl.to(device)

        # lbl_smoothed : (bsz, 2)
        lbl_smoothed = torch.stack([1-lbl_smoothed,lbl_smoothed],dim=1).squeeze(-1)
        probs = model(seq1, seq2)
        # probs : (bsz, 2)
        # criterion.weight = class_weight.to(device)
        loss = criterion(probs, lbl_smoothed)
        loss.backward()
        losses.append(loss.data.item())
        # print(sum-param_sum(model.parameters()),sum,param_sum(model.parameters()))
        # sum=param_sum(model.parameters())
        # clip_grads(model)
    optim.step()

    return np.array(losses).sum()/len(losses)

def restore_log(opt):
    basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
    json_fname = basename + ".json"
    history = json.loads(open(json_fname, "rt").read())

    return history['loss'],history['accuracy'],\
           history['precs'],history['recalls'],history['f1s']

def train(train_iter, val_iter, epoch, model,
          optim, criterion, opt, class_probs):
    # sum=param_sum(model.parameters())
    losses=[]
    accurs=[]
    f1s=[]
    precs=[]
    recalls=[]

    if opt.load_idx != -1:
        losses,accurs,\
        precs,recalls,\
        f1s=restore_log(opt.load_idx)

    epoch_start = epoch['start']
    epoch_end = epoch['end']

    valid(val_iter,model)

    samples = []
    for epoch in range(epoch_start,epoch_end):
        nbatch = 0
        for i, sample in enumerate(train_iter):
            nbatch += 1
            samples.append(sample)

            if len(samples) == opt.accum_count:
                loss_val = train_batches(samples,model,criterion,
                                   optim,class_probs,opt)

                losses.append(loss_val)
                percent = i/len(train_iter)
                progress_bar(percent,loss_val,epoch)

                samples = []

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
            # probs : (bsz, 2)
            cmp = (probs[:,0]-probs[:,1]).cpu()
            cmp.apply_(lambda x:0 if x>0 else 1)
            # pred = probs.cpu().apply_(lambda x: 0 if x < 0.5 else 1)
            pred_lst.extend(cmp.numpy())
            lbl_lst.extend(lbl.numpy())

    accurracy = metrics.accuracy_score(np.array(lbl_lst),np.array(pred_lst))
    precision = metrics.precision_score(np.array(lbl_lst),np.array(pred_lst))
    recall =metrics.recall_score(np.array(lbl_lst),np.array(pred_lst))
    f1 = metrics.f1_score(np.array(lbl_lst),np.array(pred_lst))

    return accurracy, precision, recall, f1

def dataset_weight(train_iter):
    npos = 0
    nneg = 0
    for sample in train_iter:
        b_npos = sample.lbl.numpy().sum()
        npos += b_npos
        nneg += sample.lbl.shape[0] - b_npos

    return {'wpos': 1 - npos / (npos + nneg), 'wneg': 1 - nneg / (npos + nneg)}

def class_dist(train_iter):
    npos = 0
    nneg = 0
    for sample in train_iter:
        b_npos = sample.lbl.numpy().sum()
        npos += b_npos
        nneg += sample.lbl.shape[0] - b_npos

    return {'ppos': npos / (npos + nneg), 'pneg': nneg / (npos + nneg)}

def unk_ratio(val_iter,SEQ1):
    nunk = 0
    ntotal = 0
    for sample in val_iter:
        for seq in [sample.seq1, sample.seq2]:
            nunk += seq.apply_(lambda x: 1 if x == SEQ1.vocab.stoi['<unk>'] else 0).numpy().sum()
            ntotal += sample.seq1.shape[0] * sample.seq1.shape[0]
    print(nunk / ntotal, nunk, ntotal)

def init_model(model_opt, model):
    if model_opt.param_init != 0.0:
        print('Intializing model parameters.')
        for p in model.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)

    if model_opt.param_init_glorot:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    if hasattr(model.encoder, 'embeddings'):
        model.encoder.embeddings.load_pretrained_vectors(
            model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()

    SEQ1, SEQ2,\
    train_iter, val_iter = iters.build_iters(bsz=opt.batch_size)

    # class_weight = dataset_weight(train_iter)
    class_probs = class_dist(train_iter)
    # print('Class weight: ',class_weight)
    print('Class distribution: ',class_probs)

    embeddings_enc = model_builder.build_embeddings(opt, SEQ1.vocab, [])
    encoder = enc.TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings_enc)

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    model = PhraseSim(encoder,opt).to(device)
    print('Param sum before init: ', param_sum(model.parameters()))
    init_model(opt, model)
    print('Param sum after init: ', param_sum(model.parameters()))


    # print(model.state_dict())
    if opt.load_idx != -1:
        basename = "{}-epoch-{}".format(opt.exp, opt.load_idx)
        model_fname = basename + ".model"
        location = {'cuda:'+str(opt.gpu):'cuda:'+str(opt.gpu)} if opt.gpu !=-1 else 'cpu'
        model_dict = torch.load(model_fname, map_location=location)
        model.load_state_dict(model_dict)
        print("Loading model from '%s'" % (model_fname))

    # model.generator = generator.to(device)
    optim = optimizers.build_optim(model,opt,None)
    # criterion = nn.NLLLoss()
    criterion = nn.KLDivLoss()
    epoch = {'start':opt.load_idx if opt.load_idx != -1 else 0,
             'end':10000}

    train(train_iter,val_iter,epoch,
          model,optim,criterion,opt,class_probs)

