''''TODO docstring'''

import torch
import torch.utils.data as tud
from tqdm import tqdm
from mycoai import utils

class MLMTask:
    '''Masked Language Modelling''' #TODO

    @staticmethod
    def train(model, data, epochs=100, batch_size=64, p_mask=0.15, p_random=0.1, p_unchanged=0.1, sampler=None, optimizer=None):
        
        dataloader = tud.DataLoader(data, batch_size=batch_size, sampler=sampler)

        for epoch in tqdm(range(epochs)):
            # model.train()
            for x, _ in dataloader:
                full_mask = torch.randn(x.shape) < p_mask
                for token in utils.TOKENS.values():
                    full_mask = full_mask & (x != token)
                random_mask = full_mask & (torch.randn(x.shape) < p_random)
                unchanged_mask = full_mask % (torch.randn(x.shape) < p_unchanged)
                print(full_mask)
                print(x)
                exit()

class NSPTask:
    '''Next Sentence Prediction''' #TODO

    def __init__(self):
        pass

    def train(self, data):
        pass