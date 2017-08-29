#!/usr/bin/env python

"""
    lifted-model.py
    
    Metric learning using `lifted_loss`
"""


from __future__ import division

import sys
import argparse
import numpy as np
import pandas as pd
import ujson as json

from time import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('./losses')
from pytorch_lifted_loss import lifted_loss

# --
# Define model

class CharacterLSTM(nn.Module):
    
    def __init__(self, n_chars, n_classes, emb_dim=64, hidden_dim=64):
        super(CharacterLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.char_embs = nn.Embedding(n_chars, emb_dim)
        self.lstm      = nn.LSTM(emb_dim, hidden_dim)
        self.fc1       = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        embeds = self.char_embs(x)
        self.lstm_out, _ = self.lstm(embeds.view(embeds.size(0), 1, -1))
        return self.fc1(self.lstm_out[-1,0,:].view(1, -1))


# --
# Setup

seed = 123
epochs = 20

_ = torch.manual_seed(seed)
if torch.cuda.is_available():
    _ = torch.cuda.manual_seed(seed)

# --
# IO

data = map(json.loads, open('./data.jl'))
X, y = zip(*[d.values() for d in data])
X, y = np.array(X), np.array(y)

uchars = set(reduce(lambda a,b: a+b, X))
char_lookup = dict(zip(uchars, range(len(uchars))))

ulabels = set(y)
label_lookup = dict(zip(ulabels, range(len(ulabels))))

X_ten = [torch.LongTensor([char_lookup[xx] for xx in x]).cuda() for x in X]
y_ten = [torch.LongTensor([label_lookup[yy]]).cuda() for yy in y]

ten = zip(X_ten, y_ten)
ten_train, ten_test = train_test_split(ten, train_size=0.8, random_state=123)
del ten

# --
# Define model

lstm_params = {
    "n_chars"    : len(char_lookup),
    "n_classes"  : len(label_lookup),
    "emb_dim"    : 16,
    "hidden_dim" : 32
}

model = CharacterLSTM(**lstm_params)
model.cuda()
opt = torch.optim.Adam(model.parameters())

# --
# Train

log_interval = 100
batch_size = 10

_ = model.train()
for epoch in range(2):
    total_loss = 0
    res = 0
    epoch_start_time = time()
    
    inds = np.arange(len(ten_train))
    inds = np.random.choice(inds, inds.shape[0], replace=False)
    inds = inds.reshape((-1, batch_size))
    
    counter = 0
    for ind in inds:
        
        model.zero_grad()
        
        score = torch.cat([model(Variable(ten_train[j][0])) for j in ind])
        target = torch.cat([Variable(ten_train[j][1]) for j in ind])
        
        loss = lifted_loss(score, target)
        loss.backward()
        opt.step()
        
        # Logging
        total_loss += loss.data[0]
        counter += batch_size
        if not counter % log_interval:
            print "Epoch=%d | i=%d | Loss=%f | Epoch Time=%f" % (
                epoch, 
                counter, 
                total_loss / log_interval, 
                time() - epoch_start_time,
            )
            total_loss = 0



model.eval()
preds = torch.cat([model(Variable(x)) for x,_ in ten_test]).cpu().data.numpy()
act = torch.cat([y for _,y in ten_test]).cpu().numpy()
