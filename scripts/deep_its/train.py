import argparse
import torch
from mycoai import utils
from mycoai.data import Data
from mycoai.deep.models import DeepITSClassifier, ResNet, BERT
from mycoai.deep.train import DeepITSTrainer

def train(fasta_filepath, base_arch_type, valid_split, epochs, weighted_loss, weighted_sampling, learning_rate, weight_decay):
    '''TODO'''

    # Inferring dna_encoder from base arch
    if base_arch_type == 'BERT':
        dna_encoder = 'bpe'
    if base_arch_type == 'ResNet':
        dna_encoder = '4d'

    # Reading/encoding the data
    data = Data(fasta_filepath)
    if valid_split > 0:
        train_data, valid_data = data.encode_dataset(dna_encoder, valid_split)
    else:
        train_data = data.encode_dataset(dna_encoder, valid_split)
        valid_data = None

    # Defining model
    if base_arch_type == 'BERT':
        base_arch = BERT(train_data.dna_encoder.len_input, 
                         train_data.dna_encoder.vocab_size)
        fcn = []
    else:
        base_arch = ResNet([2,2,2,2])
        fcn = [] # TODO
    model = DeepITSClassifier(base_arch, train_data.dna_encoder, 
                              train_data.tax_encoder, fcn, output_head)
    
    # Training
    sampler = train_data.weighted_sampler() if weighted_sampling else None
    loss = None
    if weighted_loss:
        loss = train_data.weighted_loss(torch.nn.CrossEntropyLoss, sampler)
    optimizer = torch.optim.Adam(lr=learning_rate, wd=weighted_loss)
    model, history = DeepITSTrainer.train(model, train_data, valid_data, epochs,
                                          loss, batch_size, sampler, optimizer)
    torch.save(model, save_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python train.py',
        description='Trains a deep neural network for taxonomic classification \
            of fungal ITS sequences.')
    
    parser.add_argument('fasta_filepath',
        help='Path to the FASTA file containing ITS sequences for training.')
    
    parser.add_argument('--save_model', '-m', 
        default='models/test2.pt',
        type=str,
        nargs=1,
        help="Path to saved DeepITSClassifier Pytorch model.")
    
    args = parser.parse_args()
    train()