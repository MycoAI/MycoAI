'''Contains weight schedulers that assign a weight to the loss for each of 
the taxon levels, depending on the current training epoch.'''

import torch
from mycoai import utils

class Constant:
    '''Constant taxon level weights throughout all training epochs.'''

    def __init__(self, weights):
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, epoch):
        return self.weights
    
    def get_config(self):
        return {'type':    utils.get_type(self),
                'weights': self.weights}

class Curriculum:
    '''Gradually increases task difficulty/specifity.
    
    Parameters
    ----------
    thresholds: list[float]
        List of length 6, specifying training progress fraction at which the 
        corresponding taxon level is added (first entry is always 0).
    low: float
        Weight for taxon levels not within current curriculum (default is 0.0)
    high: float
        Weight for taxon levels within current curriculum (default is 1.0)
    maintain: bool
        If True, keep weights for higher levels than current (default is False)
    '''

    def __init__(self, thresholds, n_epochs, low=0.0, high=1.0, maintain=False):
        self.thresholds = thresholds
        self.n_epochs = n_epochs
        self.low = low 
        self.high = high 
        self.maintain = maintain
        self.i = 0

    def __call__(self, epoch):
        for lvl in range(len(self.thresholds)):
            if epoch/self.n_epochs > self.thresholds[lvl]:
                self.i = lvl
        if self.maintain:
            weights = ([self.high]*(1+self.i) + 
                       [self.low]*(len(self.thresholds)-self.i-1))
        else:
            weights = ([self.low]*self.i + [self.high] + 
                       [self.low]*(len(self.thresholds)-1-self.i))
        return torch.tensor(weights, dtype=torch.float32)

    def get_config(self):
        return {'type':       utils.get_type(self),
                'thresholds': self.thresholds}

class Alternating:
    '''Focus on a different taxon level with each epoch'''

    def __init__(self, low=0.0, high=1.0, n_levels=6):
        self.low = low 
        self.high = high
        self.n_levels = n_levels

    def __call__(self, epoch):
        i = epoch % self.n_levels
        weights = [self.low]*i + [self.high] + [self.low]*(self.n_levels-1-i)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_config(self):
        return {'type': utils.get_type(self)}