"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from preproc.iters import PAD_WORD
import onmt.modules
from onmt.encoders.transformer import TransformerEncoder
from onmt.modules import Embeddings, CopyGenerator
from onmt.utils.misc import use_gpu


def build_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam")
