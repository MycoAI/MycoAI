'''Contains modules that inherit from torch.nn.Module, the most important one
being the SeqClassNetwork class.'''

from mycoai.modules.seq_class_network import SeqClassNetwork
from mycoai.modules.cnns import SimpleCNN, ResNet
from mycoai.modules.transformers import BERT, EncoderDecoder