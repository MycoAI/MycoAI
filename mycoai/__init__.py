import torch
from . import utils

# Set up device (GPU if available)
if torch.cuda.is_available():
    utils.DEVICE = torch.device('cuda')
else:
    utils.DEVICE = torch.device('cpu')