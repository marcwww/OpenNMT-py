import argparse
import onmt.opts as opts
from preproc import iters
from onmt import model_builder
from onmt.utils import optimizers
from itertools import count
from onmt.encoders import transformer_ps as enc
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

LOGGER = logging.getLogger(__name__)
SAVE_PER = 10

def clip_grads(model):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

class Dense(nn.Module):

    def __init__(self, max_length, rnn_size):
        super(Dense, self).__init__()
        self.linear = nn.Linear(max_length*rnn_size,1)
        padding_base = torch.zeros(1)
        self.register_buffer('padding_base',padding_base)
        self.max_length = max_length
        self.rnn_size = rnn_size

    def forward(self, memory_bank):
        # memory_bank: (seq_len, bsz, hdim)
        seq_len, bsz, hdim = memory_bank.shape
        padding_base = self.padding_base.\
            expand(self.max_length,bsz,self.rnn_size).\
            clone()
        # mem_padded: (max_len, bsz, hdim)
        padding_base[:seq_len,:,:] += memory_bank
        mem_padded = padding_base
        # mem_transpose: (bsz, max_len, hdim)
        mem_transpose = mem_padded.transpose(0,1).contiguous()
        # mem_flatten: (bsz, max_len*hdim)
        mem_flatten = mem_transpose.view(bsz,-1)

        # return: (bsz,1)
        return self.linear(mem_flatten)

class PhraseSim(nn.Module):

    def __init__(self, encoder, opt):
        super(PhraseSim, self).__init__()
        self.encoder = encoder
        self.generator = nn.Sequential(
            Dense(opt.max_len_total,opt.rnn_size),
            nn.Sigmoid())
        # self.generator = nn.Sequential(
        #     nn.Linear(opt.rnn_size, 1),
        #     nn.Sigmoid())

    def forward(self, seq1, seq2, device):
        seq1 = seq1.unsqueeze(2)
        seq2 = seq2.unsqueeze(2)

        enc_final, memory_bank, src = self.encoder(seq1, seq2)

        return memory_bank

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

def train_batch(sample, model, criterion, optim):
    model.train()

    model.zero_grad()
    seq1, seq2, lbl = sample.seq1, \
                      sample.seq2, \
                      sample.lbl

    seq1 = seq1.to(device)
    seq2 = seq2.to(device)
    lbl = lbl.type(torch.FloatTensor).to(device)

    # seq : (seq_len,bsz)
    # lbl : (bsz)
    memory_bank = model(seq1, seq2, device)
    # decoder_outputs : (1,bsz,hdim)
    # probs : (bsz)
    probs = model.generator(memory_bank).squeeze(1)
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

def train(train_iter, val_iter, epoch, model, optim, criterion, opt):
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

    for epoch in range(epoch_start,epoch_end):
        nbatch = 0
        for i, sample in enumerate(train_iter):
            nbatch += 1

            loss = train_batch(sample,model,criterion,optim)

            loss_val = loss.data.item()
            losses.append(loss_val)
            percent = i/len(train_iter)
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

def valid(val_iter,model):
    model.eval()

    nt = 0
    nc = 0
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
            memory_bank = model(seq1, seq2, device)
            # decoder_outputs : (1,bsz,hdim)
            # probs : (bsz)
            probs = model.generator(memory_bank).squeeze(1)
            pred = probs.cpu().apply_(lambda x: 0 if x < 0.5 else 1)
            pred_lst.extend(pred.numpy())
            lbl_lst.extend(lbl.numpy())

            nw = (probs-lbl).apply_(lambda x: 0 if x == 0 else 1).numpy().sum()

            bsz = lbl.shape[0]
            nc += (bsz-nw)
            nt += bsz

    accurracy = metrics.accuracy_score(np.array(lbl_lst),np.array(pred_lst))
    precision = metrics.precision_score(np.array(lbl_lst),np.array(pred_lst))
    recall =metrics.recall_score(np.array(lbl_lst),np.array(pred_lst))
    f1 = metrics.f1_score(np.array(lbl_lst),np.array(pred_lst))

    return accurracy, precision, recall, f1

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
    embeddings_enc = model_builder.build_embeddings(opt, SEQ1.vocab, [])
    encoder = enc.TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings_enc)
    decoder = dec.TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.self_attn_type,
                                  opt.dropout, opt.tgt_word_vec_size,
                                     SEQ1.vocab.stoi[iters.PAD_WORD])
    # print(param_sum(embeddings_enc.parameters()),
    #       param_sum(encoder.parameters()),
    #       param_sum(decoder.parameters()))
    # generator = nn.Sequential(
    #     nn.Linear(opt.rnn_size, 1),
    #     nn.Sigmoid())

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    model = PhraseSim(encoder,opt).to(device)
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
    criterion = nn.BCELoss(size_average=True)

    epoch = {'start':opt.load_idx if opt.load_idx != -1 else 0,
             'end':10000}

    train(train_iter,val_iter,epoch,
          model,optim,criterion,opt)

