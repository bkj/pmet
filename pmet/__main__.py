#!/usr/bin/env python

"""
    __main__.py
"""

from __future__ import print_function, division

import sys
import json
import argparse
import pandas as pd
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from pmet.models import CharLSTM, AttCharLSTM
from pmet.data import make_train_dataset, load_test_dataset

# --
# Setup

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training options
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--outpath', type=str, default='./model')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=100)
    
    # Testing options
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--data-path', type=str)
    
    # Generic options
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    # Validate parameters
    if args.attention and not args.bidirectional:
        raise Exception('if attention=1, bidirectional=1')
    
    if not torch.cuda.is_available():
        print('warning: cuda is not available!', file=sys.stderr)
        args.no_cuda = True
    
    return args

# --
# Helpers

def train_epoch(model, train_iter, opt, decay=0.9):
    _ = model.train()
    for batch_idx, batch in tqdm(enumerate(train_iter), total=len(train_iter)):
        _, _ = model.train_step(batch.val, batch.lab.squeeze(), opt)


def test(model, test_iter):
    _ = model.eval()
    for batch_idx, batch in tqdm(enumerate(test_iter), total=len(test_iter)):
        score = model(batch.val)
        pred = score.max(1)[1].data.squeeze()[0]
        
        if hasattr(batch, 'lab'):
            y = batch.lab.squeeze()
            yield {
                "pred" : pred,
                "act" : y.data.squeeze()[0],
            }
        else:
            yield {
                "pred" : pred,
            }

# --
# Train vs. test

def test_model(args):
    
    # --
    # IO
    
    assert args.data_path is not None, "args.data_path is None"
    
    LABS, VALS = pickle.load(open('model.vocab'))
    test_iter = load_test_dataset(args.data_path, VALS, device=-args.no_cuda)
    
    # --
    # Model definition
    
    attention, model_config = json.load(open(args.model_path + '.json'))
    model = AttCharLSTM(**model_config) if attention else CharLSTM(**model_config)
    model.load_state_dict(torch.load(args.model_path + '.pt'))
    
    if not args.no_cuda:
        model = model.cuda()
    
    preds = pd.DataFrame(list(test(model, test_iter)))
    preds.pred = preds.pred.apply(lambda x: LABS.vocab.itos[x])
    preds.to_csv(sys.stdout, sep='\t', index=False, header=False)


def train_model(args):
    
    # --
    # IO
    
    dataset = make_train_dataset(args.data_dir, device=-args.no_cuda)
    (train_iter, test_iter) = dataset['iterator']
    
    # --
    # Initialize model
    
    print('initializing character-level LSTM | attention=%d | bidirectional=%d' %\
        (args.attention, args.bidirectional), file=sys.stderr)
    
    if args.attention:
        model_config = {
            "n_chars"   : dataset['n_chars'],
            "n_classes" : dataset['n_classes'],
            
            "att_channels"   : 5,
            "att_dim"        : 16,
            "emb_dim"        : 64, 
            "rec_hidden_dim" : 32,
        }
        model = AttCharLSTM(**model_config)
    else:
        model_config = {
            "n_chars"   : dataset['n_chars'],
            "n_classes" : dataset['n_classes'],
            
            "bidirectional"  : args.bidirectional,
            "emb_dim"        : 64,
            "rec_hidden_dim" : 32,
        }
        model = CharLSTM(**model_config)
    
    if not args.no_cuda:
        model = model.cuda()
    
    # --
    # Train model
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    
    for epoch in range(args.epochs):
        print('epoch=%d | train' % epoch, file=sys.stderr)
        _ = train_epoch(model, train_iter, opt)
        
        print('epoch=%d | test' % epoch, file=sys.stderr)
        preds = pd.DataFrame(list(test(model, test_iter)))
        print('acc=%f\n--' % (preds.act == preds.pred).mean(), file=sys.stderr)
    
    # --
    # Save model
    
    print("saving model to %s" % args.outpath, file=sys.stderr)
    torch.save(model.state_dict(), args.outpath + '.pt')
    json.dump((args.attention, model_config), open(args.outpath + '.json', 'w'))
    pickle.dump(dataset['vocabs'], open(args.outpath + '.vocab', 'w'))


# --
# Evaluation

if __name__ == "__main__":
    args = parse_args()
    if args.model_path is not None:
        print('pmet: testing', file=sys.stderr)
        test_model(args)
    else:
        print('pmet: training', file=sys.stderr)
        train_model(args)
