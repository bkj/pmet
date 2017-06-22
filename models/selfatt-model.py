#!/usr/bin/env python

"""
    selfatt-model.py
    
    From https://openreview.net/pdf?id=BJC_jUqxe
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

class ACharacterLSTM(nn.Module):
    """ Character LSTM w/ attention """
    def __init__(self, n_chars, n_classes, emb_dim=64, rec_hidden_dim=32, att_dim=30, att_channels=16):
        super(ACharacterLSTM, self).__init__()
        
        self.char_embs = nn.Embedding(n_chars, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, int(rec_hidden_dim / 2), bias=False, bidirectional=True)
        
        self.att1 = nn.Linear(rec_hidden_dim, att_dim, bias=False)
        self.att2 = nn.Linear(att_dim, att_channels, bias=False)
        
        self.fc1 = nn.Linear(att_channels * rec_hidden_dim, n_classes)
        
        self.I = Variable(torch.eye(att_channels))
    
    def _encode(self, x):
        # one-hot -> biLSTM encoded
        x = self.char_embs(x)
        x = x.view(x.size(0), 1, -1)
        x, _ = self.rnn(x)
        x = x[:,0,:]
        return x
    
    def _attention(self, x):
        A = F.softmax(self.att2(F.tanh(self.att1(x))).t())
        x = torch.mm(A, x)
        x = x.view(1, -1)
        return x, A
    
    def _penalty(self, A):
        return torch.norm(torch.mm(A, A.t()) - self.I) ** 2
    
    def forward(self, x):
        x = self._encode(x)
        x, A = self._attention(x)
        p = self._penalty(A)
        
        self.A, self.p = A, p
        
        return self.fc1(x).view(-1)

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

lambda_ = 1

model = ACharacterLSTM(**{
    "n_chars"    : len(char_lookup) + 1,
    "n_classes"  : len(label_lookup)
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
        loss = loss_function(score.view(1, -1), Variable(yt)) + lambda_ * model.p
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
