#!/usr/bin/env python

"""
    pytorch_dlib_metric_loss.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


def dlib_metric_loss(score, target, margin=1, extra_margin=0, cuda=False):
    loss = Variable(torch.FloatTensor([0]))
    if cuda:
        loss = loss.cuda()
    
    bsz = score.size(0)
    
    # Compute distance matrix
    mag  = (score ** 2).sum(1).expand(bsz, bsz)
    sim  = score.mm(score.transpose(0, 1))
    dist = (mag + mag.transpose(0, 1) - 2 * sim)
    dist = torch.nn.functional.relu(dist).sqrt()
    
    # Determine number of positive + negative thresh
    neg_mask = target.expand(bsz, bsz)
    neg_mask = (neg_mask - neg_mask.transpose(0, 1)) != 0
    
    n_pos = (1 - neg_mask).sum() # Number of pairs
    n_pos = (n_pos - bsz) / 2 # Number of pairs less diagonal, w/o repetition
    n_pos = n_pos.data[0]
    
    neg_thresh = dist[neg_mask].sort()[0][n_pos].data[0]
        
    for r in range(bsz):
        x_label = target[r].data[0]
        
        for c in range(bsz):
            y_label = target[c].data[0]
            
            if r == c:
                continue
            
            d = dist[r,c]
            if x_label == y_label:
                # Positive examples should be less than (margin - extra_margin)
                if d.data[0] > margin - extra_margin:
                    loss += d - (margin - extra_margin)
            else:
                # Negative examples should be greater than (margin + extra_margin)
                # But... we'll only use the hardest negative pairs
                if (d.data[0] < margin + extra_margin) and (d.data[0] < neg_thresh):
                    loss += (margin + extra_margin) - d
    
    return loss / (2 * n_pos)


# --

if __name__ == "__main__":
    import numpy as np
    np.random.seed(123)
    
    score = np.random.uniform(0, 1, (20, 3))
    target = np.random.choice(range(3), 20)
    
    score = Variable(torch.FloatTensor(score))
    target = Variable(torch.LongTensor(target))
    
    dlib_metric_loss(score, target)
