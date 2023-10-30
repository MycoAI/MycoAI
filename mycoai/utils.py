'''Contains constants and helper functions to support DeathCap.'''

import torch
import warnings
import wandb
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
MAX_LEN = 5000 # Max length of positional encodings transformers
# NOTE Be careful with changing this one: BPE assumes TOKENS['MASK'] == 0
TOKENS = {'MASK':0, 'CLS':1, 'SEP':2, 'PAD':3, 'UNK':4}

filename_from_path = lambda path: path.split('/')[-1].split('\\')[-1]

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

def get_type(object):
    '''Get type of object as str without including parent modules'''
    return str(type(object)).split('.')[-1][:-2]

def get_config(object, prefix=''):
    '''Gets configuration data from object, if available'''
    if prefix == 'opt':
        config = get_opt_config(object)
    elif prefix == 'sampler':
        config = get_sampler_config(object)
    elif prefix == 'loss':
        config = get_loss_config(object)
    else:
        config = getattr(object, 'get_config', lambda: {})()
    prefix = prefix + int(len(prefix)>0)*'_'
    return {prefix + key: value for key, value in config.items()}

def get_opt_config(optimizer):
    '''Gets configuration data from torch.nn.optimizer object'''
    opt_hyperpars = optimizer.state_dict()['param_groups']
    a = int(len(opt_hyperpars)>1) # Check if there is more than one param group
    opt_config = {'type': get_type(optimizer)}
    for i, group in enumerate(opt_hyperpars):
        opt_config.update(
            {f'gr{i}_'*a + 'lr': group.get('lr', '?'),
             f'gr{i}_'*a + 'betas':        group.get('betas', '?'),
             f'gr{i}_'*a + 'weight_decay': group.get('weight_decay', '?')}
        )
    return opt_config

def get_sampler_config(sampler):
    '''Gets configuration data from sampler object'''
    sampler_config = {}
    if hasattr(sampler, 'lvl'):
        sampler_config.update({'lvl': sampler.lvl})
    if hasattr(sampler, 'unknown_frac'):
        sampler_config.update({'unknown_frac': sampler.unknown_frac})
    return {'type': get_type(sampler), **sampler_config}

def get_loss_config(loss):
    '''Gets configuration data from loss object'''
    loss_config = {'weighted': hasattr(loss, 'weighted')}
    if hasattr(loss, 'sampler_correction'):
        loss_config.update({'sampler_correction': loss.sampler_correction})
    return {'type': get_type(loss), **loss_config}