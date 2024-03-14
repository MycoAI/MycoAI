'''Restores a DeepITSClassifier model that was saved in an earlier version of 
MycoAI, by converting it to the current SeqClassNetwork class.

Only works with old mycoai/deep/models folder in place. This folder can 
be downloaded from https://github.com/MycoAI/MycoAI/tree/aed05bbff79608ff0cbcc216611a02f3b7a02d72/mycoai/deep/models'''

import torch
from mycoai import utils
from mycoai.modules import SeqClassNetwork
from mycoai.modules import BERT, SimpleCNN

# Update this variable for desired model
old_model_path = '/data/luuk/models/MycoAI-Vu-CNN LS 0.02 0.02 0.02 0.02 0.02 0.0.pt'
new_model_path = '/data/luuk/models/MycoAI-CNN-HLS.pt'

# Load as DeepITSClassifier object and extract information
model = torch.load(old_model_path, map_location=utils.DEVICE)
state_dict = model.state_dict() # Extract network weights
dna_encoder = model.dna_encoder # Encoders stayed the same after update...
tax_encoder = model.tax_encoder # ... so we can get them directly
config = model.get_config()
base_arch_type = config['base_arch_type'] # Architectures moved...
if base_arch_type == 'BERT': # ... so we must reinitialize them
    arch = BERT(
        config['dna_encoder_vocab_size'], config['base_arch_d_model'],
        config['base_arch_d_ff'], config['base_arch_h'], config['base_arch_N']
    )
elif base_arch_type == 'SimpleCNN':
    arch = SimpleCNN(conv_layers=config['base_arch_layers'])
else:
    raise RuntimeError('Please initialize the desired architecture here.')

# Create SeqClassNetwork object
output = 'multi' if config['output_type'] == 'MultiHead' else 'infer_sum'
model = SeqClassNetwork(arch, dna_encoder, tax_encoder, config['fcn'], 
                        output=output)
print(model.load_state_dict(state_dict))
model.eval() # Put in evaluation mode (for safety) before saving

torch.save(model, new_model_path)
print("Success.")