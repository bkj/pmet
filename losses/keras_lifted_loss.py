#!/usr/bin/env python

"""
    keras_lifted_loss.py
"""

from keras import backend as K

def lifted_loss(margin=1):
    """
      Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
      Implemented in `keras`
        
      See also the `pytorch` implementation at: https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4
    """
    def f(target, score):
        
        # Compute mask (-1 for different class, 1 for same class, 0 for diagonal)
        mask = (2 * K.equal(0, target - K.reshape(target, (-1, 1))) - 1)
        mask = (mask - K.eye(score.shape[0]))
        
        # Compute distance between rows
        mag  = (score ** 2).sum(axis=-1)
        mag  = K.tile(mag, (mag.shape[0], 1))
        dist = (mag + mag.T - 2 * score.dot(score.T))
        dist = K.sqrt(K.maximum(0, dist))
        
        # Negative component (points from different class should be far)
        l_n = K.sum((K.exp(margin - dist) * K.equal(mask, -1)), axis=-1)
        l_n = K.tile(l_n, (score.shape[0], 1))
        l_n = K.log(l_n + K.transpose(l_n))
        l_n = l_n * K.equal(mask, 1)
        
        # Positive component (points from same class should be close)
        l_p = dist * K.equal(mask, 1)
        
        loss  = K.sum((K.maximum(0, l_n + l_p) ** 2))
        n_pos = K.sum(K.equal(mask, 1))
        loss /= (2 * n_pos)
        
        return loss
        
    return f

# --

if __name__ == "__main__":
    import numpy as np
    np.random.seed(123)
    
    score = np.random.uniform(0, 1, (20, 3))
    target = np.random.choice(range(3), 20)
    
    print lifted_loss(1)(target, score).eval()