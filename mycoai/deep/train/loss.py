'''Cross-entropy loss function for MycoAI deep ITS classifier. Supports input
types required for MycoAI yet unsupported by default PyTorch.'''

import torch
from mycoai import utils

class CrossEntropyLoss(torch.nn.Module):
    '''Like torch.nn.CrossEntropyLoss, but expects input as probability 
    distribution (softmax-activated). Supports ignore_index even if target is 
    probability distribution.'''

    def __init__(self, weight=None, ignore_index=-100):
        '''Initializes CrossEntropyLoss. Allows specification of class weights
        and a class index to be ignored.'''

        super().__init__()
        self.ignore_index = ignore_index
        if weight is not None:
            self.weight = weight.to(utils.DEVICE)
            self.forward = self._forward_weighted
        else:
            self.forward = self._forward

    def _forward(self, input, target, target_index):
        '''Calculate cross-entropy loss. Input and target must be probabilities.
        Target_index must be tensor containing class indices.'''

        w = (target_index != self.ignore_index).unsqueeze(1)
        return -torch.sum(w*target * torch.log(input + 1e-10)) / w.sum()
    
    def _forward_weighted(self, inputt, target, target_index):
        '''Calculate weighted cross-entropy loss. Input and target must be 
        probabilities. Target_index must be tensor containing class indices.'''

        dont_ignore = target_index != self.ignore_index
        w = (self.weight[torch.where(dont_ignore, target_index, 0)]*
             (dont_ignore)).unsqueeze(1)
        
        return -torch.sum(w*target*torch.log(inputt + 1e-7))/(w.sum() + 1e-7)