#!/usr/bin/env python

"""
    bilstm-model.py
    
    Typical biLSTM model
"""

from __future__ import division

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import ujson as json
import numpy as np
import pandas as pd
from time import time

# --
# Define model

class CharacterLSTM(nn.Module):
    """ Character LSTM """
    def __init__(self, n_chars, n_classes, emb_dim=64, rec_hidden_dim=32, bidirectional=True):
        super(CharacterLSTM, self).__init__()
        
        self.char_embs = nn.Embedding(n_chars, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, int(rec_hidden_dim / (1 + bidirectional)), bias=False, bidirectional=bidirectional)
        
        self.fc1 = nn.Linear(rec_hidden_dim, n_classes)
    
    def forward(self, x):
        x = self.char_embs(x)
        x = x.view(x.size(0), 1, -1)
        x, _ = self.rnn(x)
        x = x[:,0,:]
        x = self.fc1(x)
        return x[-1]

# --
# Setup

seed = 123
epochs = 20

_ = torch.manual_seed(seed)

# --
# IO

def load_data(path):
    data = map(json.loads, open(path))
    X, y = zip(*[d.values() for d in data])
    X, y = np.array(X), np.array(y)
    return X, y, data

X_train, y_train, train_data = load_data('./data/train.jl')

uchars = set(reduce(lambda a,b: a+b, X_train))
char_lookup = dict(zip(uchars, range(1, len(uchars) + 1)))

ulabels = set(y_train)
label_lookup = dict(zip(ulabels, range(len(ulabels))))
rev_label_lookup = dict(zip(range(len(ulabels)), ulabels))

X_train = np.array([torch.LongTensor([char_lookup[xx] for xx in x]) for x in X_train])
y_train = np.array([torch.LongTensor([label_lookup[yy]]) for yy in y_train])

# --
# Define model

model = CharacterLSTM(**{
    "n_chars"    : len(char_lookup) + 1,
    "n_classes"  : len(label_lookup),
    
    "emb_dim" : 128,
    "rec_hidden_dim" : 256,
})

loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

# --
# Train

log_interval = 100

model.train()
for epoch in range(1):
    total_loss = 0
    res = 0
    epoch_start_time = time()
    p = np.random.permutation(X_train.shape[0]) # shuffle = True
    for i,(xt, yt) in enumerate(zip(X_train[p], y_train[p])):
        
        # Training
        model.zero_grad()
        score = model(Variable(xt))
        loss = loss_function(score.view(1, -1), Variable(yt))
        loss.backward()
        opt.step()
        
        # Logging
        total_loss += loss.data[0]
        res += yt[0] == score.max(0)[1].data[0]
        if i and not i % log_interval:
            print "Epoch=%d | i=%d | Loss=%f | Epoch Time=%f | Correct=%d" % (
                epoch, 
                i, 
                total_loss / log_interval, 
                time() - epoch_start_time,
                res
            )
            total_loss = 0
            res = 0

# --
# Evaluation

model.eval()

X_test, y_test, test_data = load_data('./data/test.jl')
X_test = np.array([torch.LongTensor([char_lookup.get(xx, 0) for xx in x]) for x in X_test])

test_preds = np.array([model(Variable(x)).max(0)[1].data[0] for x in X_test])
test_preds = np.array([rev_label_lookup[p] for p in test_preds])

print pd.crosstab(test_preds, y_test)
print "\n acc=%f" % (test_preds == y_test).mean()
