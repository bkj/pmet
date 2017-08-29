#!/usr/bin/env python

"""
    __main__.py
"""

from __future__ import division
from __future__ import print_function

import sys
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pmet.models import CharLSTM, AttCharLSTM
from pmet.data import make_iter

# --
# Setup

seed = 123
_ = torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()

# --
# Helpers

def train_epoch(model, train_iter, opt, decay=0.9):
    _ = model.train()
    acc = 0.0
    for batch_idx, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        
        # Prep data
        x, y = batch.val, batch.lab.squeeze()
        
        # Train step
        _, _ = model.train_step(x, y, opt)
        
        # !! Add logging


def test(model, test_iter):
    _ = model.eval()
    for batch_idx, batch in tqdm(enumerate(test_iter), total=len(test_iter)):
        x, y = batch.val, batch.lab.squeeze()
        score = model(x)
        
        pred = score.max(1)[1].data.squeeze()[0]
        
        yield {
            "pred" : pred,
            "act" : y.data.squeeze()[0],
        }

# --
# Evaluation

if __name__ == "__main__":
    
    args = parse_args()
    
    # Model definition
    if args.attention and not args.bidirectional:
        raise Exception('if attention=1, bidirectional=1')
    
    # IO
    (train_iter, test_iter), (n_chars, n_classes) = make_iter(args.data_dir, device=-args.no_cuda)
    
    print('initializing character-level LSTM | attention=%d | bidirectional=%d' %\
        (args.attention, args.bidirectional), file=sys.stderr)
    
    if args.attention:
        
        model = AttCharLSTM(**{
            "n_chars"    : n_chars,
            "n_classes"  : n_classes,
            
            "att_channels"   : 5, # !! Allow these to be passed via CLI
            "att_dim"        : 16,
            "emb_dim"        : 64, 
            "rec_hidden_dim" : 32,
        })
    else:
        model = CharLSTM(**{
            "n_chars"    : n_chars,
            "n_classes"  : n_classes,
            
            "bidirectional"  : args.bidirectional,
            "emb_dim"        : 64,
            "rec_hidden_dim" : 32,
        })
    
    if not args.no_cuda:
        model = model.cuda()
    
    print(model)
    
    opt = torch.optim.Adam(model.parameters())
    
    for epoch in range(args.epochs):
        print('epoch=%d | train' % epoch, file=sys.stderr)
        _ = train_epoch(model, train_iter, opt)
        
        print('epoch=%d | test' % epoch, file=sys.stderr)
        preds = pd.DataFrame(list(test(model, test_iter)))
        print('acc=%f\n--' % (preds.act == preds.pred).mean(), file=sys.stderr)
