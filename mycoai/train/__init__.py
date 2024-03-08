'''For training neural networks to perform taxonomic classification of 
biological sequences.'''

import wandb
from .seq_class_trainer import SeqClassTrainer
from .masked_language_modeling import MLMTrainer
from .label_smoothing import LabelSmoothing
from .loss import CrossEntropyLoss

class LrSchedule:
    '''Linearly increases the learning rate for the first warmup_steps, then
    then decreases the learning rate proportionally to 1/sqrt(step_number)'''

    def __init__(self, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_lr(self, step):
        if step == 0:
            step = 1
        return (self.d_model**(-0.5) * 
                min(step**(-0.5), step * self.warmup_steps ** (-1.5)))


class DummyScaler:
    '''A dummy gradient scaler that does not do anything and serves as a
    placeholder for when mixed precision is not desired.'''

    def __init__(self):
        pass

    def scale(self, loss):
        '''Identity function'''
        return loss

    def unscale_(self, optimizer):
        '''Empty function'''
    
    def step(self, optimizer):
        '''Optimizer step'''
        optimizer.step()

    def update(self):
        '''Empty function'''