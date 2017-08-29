
"""
    models.py
"""

from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class CharLSTM(nn.Module):
    """ bidirectional character-level LSTM """
    def __init__(self, n_chars, n_classes, emb_dim=64, rec_hidden_dim=32, bidirectional=True):
        super(CharLSTM, self).__init__()
        
        self.char_embs = nn.Embedding(n_chars, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, int(rec_hidden_dim / (1 + bidirectional)), bias=False, bidirectional=bidirectional)
        
        self.fc1 = nn.Linear(rec_hidden_dim, n_classes)
    
    def forward(self, x):
        assert(x.size(1) == 1)
        
        x = self.char_embs(x)
        x, _ = self.rnn(x)
        return self.fc1(x[-1])
    
    def train_step(self, x, y, opt):
        self.zero_grad()
        
        # Forward pass
        score = self(x)
        
        # Classification loss
        loss = F.cross_entropy(score, y)
        
        loss.backward()
        opt.step()
        
        return score, loss


class AttCharLSTM(nn.Module):
    """ bidirectionaly character-level LSTM w/ self-attention """
    def __init__(self, n_chars, n_classes, emb_dim=64, rec_hidden_dim=32, att_dim=30, att_channels=16):
        super(AttCharLSTM, self).__init__()
        
        self.char_embs = nn.Embedding(n_chars, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, int(rec_hidden_dim / 2), bias=False, bidirectional=True)
        
        self.att1 = nn.Linear(rec_hidden_dim, att_dim, bias=False)
        self.att2 = nn.Linear(att_dim, att_channels, bias=False)
        
        self.fc1 = nn.Linear(att_channels * rec_hidden_dim, n_classes)
        
        self.I = Variable(torch.eye(att_channels), requires_grad=False)
        
    
    def _encode(self, x):
        # one-hot -> biLSTM encoded
        x = self.char_embs(x)
        x, _ = self.rnn(x)
        x = x[:,0,:]
        return x
    
    def _attention(self, x):
        A = F.softmax(self.att2(F.tanh(self.att1(x))).t())
        x = torch.mm(A, x)
        return x, A
    
    def _penalty(self, A):
        return torch.norm(torch.mm(A, A.t()) - self.I) ** 2
    
    def forward(self, x):
        assert(x.size(1) == 1)
        
        x = self._encode(x)
        x, A = self._attention(x)
        p = self._penalty(A)
        
        self.A, self.p = A, p
        
        return self.fc1(x.view(1, -1))
    
    def train_step(self, x, y, opt, r_lambda=1.0):
        self.zero_grad()
        
        # Forward pass
        score = self(x)
        
        # Classification loss
        c_loss = F.cross_entropy(score, y)
        # Regularization
        r_loss = r_lambda * self.p
        # Total loss
        loss = c_loss + r_loss
        
        loss.backward()
        opt.step()
        
        return score, loss
    