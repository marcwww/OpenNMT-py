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

LOGGER = logging.getLogger(__name__)
SAVE_PER = 10

def clip_grads(model):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

class PhraseSim(nn.Module):

    def __init__(self, encoder, decoder, TGT):
        super(PhraseSim, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.TGT = TGT


    def forward(self, seq1, seq2, device):
        seq1 = seq1.unsqueeze(2)
        seq2 = seq2.unsqueeze(2)

        enc_final, memory_bank, src = self.encoder(seq1, seq2)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)

        bsz = train_iter.batch_size

        tgt = torch.LongTensor([self.TGT.vocab.stoi[BOS_WORD]]).expand(1, bsz, 1)
        tgt = tgt.to(device)

        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                    enc_state)

        return decoder_outputs, dec_state, attns


def progress_bar(percent, last_loss):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), percent, last_loss), end='')

def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')

def save_checkpoint(model, epoch, name='atec'):
    progress_clean()

    basename = "{}-epoch-{}".format(name,epoch)
    model_fname = basename + ".model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(model.state_dict(), model_fname)

def train(train_iter, val_iter, nepoches, model, optim, criterion, device):
    for epoch in range(nepoches):
        for i, sample in enumerate(train_iter):
            seq1, seq2, lbl = sample.seq1,\
                              sample.seq2,\
                              sample.lbl

            seq1 = seq1.to(device)
            seq2 = seq2.to(device)
            lbl = lbl.type(torch.FloatTensor).to(device)

            # seq : (seq_len,bsz)
            # lbl : (bsz)
            decoder_outputs, _, _ = model(seq1,seq2,device)
            # decoder_outputs : (1,bsz,hdim)
            decoder_output = decoder_outputs.squeeze(0)
            # probs : (bsz)
            probs = model.generator(decoder_output).squeeze(1)
            loss = criterion(probs,lbl)
            loss.backward()
            clip_grads(model)
            optim.step()

            loss_val = loss.data.item()
            percent = i/len(train_iter)
            progress_bar(percent,loss_val)

        if (epoch+1) % SAVE_PER == 0:
            save_checkpoint(model,epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()

    SEQ1, SEQ2, TGT,\
    train_iter, val_iter = iters.build_iters(bsz=opt.batch_size)
    embeddings_enc = model_builder.build_embeddings(opt, SEQ1.vocab, [])
    embeddings_dec = model_builder.build_embeddings(opt, TGT.vocab, [])
    encoder = enc.TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings_enc)
    decoder = dec.TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.self_attn_type,
                                  opt.dropout, embeddings_dec)
    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, 1),
        nn.Sigmoid())

    device = torch.device(opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu')

    model = PhraseSim(encoder,decoder,TGT).to(device)
    model.generator = generator.to(device)
    optim = optimizers.build_optim(model,opt,None)
    criterion = nn.BCELoss()

    train(train_iter,val_iter,1000,
          model,optim,criterion,device)

