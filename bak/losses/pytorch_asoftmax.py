#!/usr/bin/env python

"""
    pytorch_asoftmax.py
    
    Implementation of angular softmax, faithful to the original
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class CosineUnitLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        """ Linear layer that returns _angle_ between input and weights """
        super(CosineUnitLinear, self).__init__(in_features, out_features, bias=False)
    
    def forward(self, input):
        ang = self._backend.Linear.apply(F.normalize(input), F.normalize(self.weight))
        ang = ang.clamp(-1, 1).acos()
        xnorm = input.norm(p=2, dim=1)
        return ang, xnorm


def psi(x, linearized=False):
    """ 
        linearized=False -> psi from paper 
        linearized=True -> piecewise linear psi (which makes more sense to me)
    """
    if not linearized:
        ks = torch.floor(x / np.pi)
        return (1 - 2 * (ks % 2)) * x.cos() - (2 * ks)
    else:
        return torch.minimum(np.pi / 2 - x, x.cos())


def enforce_angle(ang, xnorm, target, margin=0, linearized=False):
    """ Enforce _real_ angular margin"""
    m = margin + 1 # !! Just to keep parameters consistent w/ enforce_angle
    tmp = torch.gather(ang, 1, target.view(-1, 1)).mul(m)
    ang = ang.scatter(1, target.view(-1, 1), tmp)
    ang = psi(ang, linearized)
    ang = ang.mul(xnorm.view(-1, 1).expand_as(ang))
    return ang


"""

import numpy as np
from matplotlib import pyplot as plt

s = np.arange(0, np.pi * 2, 0.01)
z = psi(torch.FloatTensor(s)).numpy()
_ = plt.plot(s, z)
plt.show()

x = torch.FloatTensor(np.random.uniform(0, 1, (20, 2)))
y = torch.LongTensor(np.random.choice((0, 1), 20))

cul = CosineUnitLinear(2, 2)
ang, xnorm = cul(Variable(x))

"""