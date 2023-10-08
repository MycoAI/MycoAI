'''Contains constants and helper functions to support DeathCap.'''

import torch
import warnings
import numpy as np
from pathlib import Path

# Constants
OUTPUT_DIR = ''
VERBOSE = 1
LEVELS = ['phylum', 'class', 'order', 'family', 'genus', 'species']
UNKNOWN_STR = '?'
UNKNOWN_INT = 9999999
PRED_BATCH_SIZE = 1000
MAX_PER_EPOCH = 500000
# NOTE Be careful with changing this one: BPE assumes TOKENS['MASK'] == 0
TOKENS = {'MASK':0, 'CLS':1, 'SEP':2, 'PAD':3, 'UNK':4}

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def set_output_dir(path, parent=''):
    '''Sets (global) output directory, creates new if it does not exist'''
    parent += min(len(parent),1)*'/'
    Path(parent + path).mkdir(parents=True, exist_ok=True)
    global OUTPUT_DIR
    OUTPUT_DIR = parent + path + '/'
    return parent + path + '/'

def set_device(name):
    '''Sets (global) PyTorch device (either `'cuda'` or `'cpu'`)'''
    global DEVICE
    if name == 'cuda' and not torch.cuda.is_available():
        name = 'cpu'
        warnings.warn('No cuda-enabled GPUs available, using CPU')
    DEVICE = torch.device(name)