'''Trains a new sequence classification neural network.'''

import argparse
import torch
from mycoai import utils
from mycoai.data import Data
from mycoai.modules import SeqClassNetwork, SimpleCNN, BERT
from mycoai.train import SeqClassTrainer
from mycoai.train.loss import CrossEntropyLoss


def train(fasta_filepath, output_filepath=utils.OUTPUT_DIR + 'model.pt', 
          base_arch_type='BERT', output='multi', hls=True, levels=utils.LEVELS, 
          epochs=50, batch_size=64, valid_split=0.1, learning_rate=0.0001, 
          weighted_loss=0.5, device=utils.DEVICE):
    '''Trains a new sequence classification neural network.''' 

    utils.set_device(device)

    # Parsing and splitting the data
    data = Data(fasta_filepath)
    train_data, valid_data = data.train_valid_split(valid_split)

    # Initializing architecture and encoding the data in correct format
    if base_arch_type == 'CNN':
        train_data = train_data.encode_dataset('kmer-spectral')
        valid_data = valid_data.encode_dataset(train_data.dna_encoder, 
                                               train_data.tax_encoder)
        arch = SimpleCNN()
        fcn_layers = [256]
    else: # base_arch_type == 'BERT'
        train_data = train_data.encode_dataset('bpe')
        valid_data = valid_data.encode_dataset(train_data.dna_encoder, 
                                               train_data.tax_encoder)
        arch = BERT(train_data.dna_encoder.vocab_size)
        fcn_layers = []

    # Initializing the actual model
    model = SeqClassNetwork(
        base_arch=arch, dna_encoder=train_data.dna_encoder, output=output,
        tax_encoder=train_data.tax_encoder, fcn_layers=fcn_layers, 
    )

    # Training
    loss = train_data.weighted_loss(CrossEntropyLoss, strength=weighted_loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ls = [0.02,0.02,0.02,0.02,0.02,0] if hls else [0,0,0,0,0,0.1]
    if levels[0][0].isnumeric():
        levels = [float(lvl) for lvl in levels]
    model, history = SeqClassTrainer.train(
        model, train_data, valid_data, epochs=epochs, loss=loss, 
        batch_size=batch_size, optimizer=optimizer, levels=levels, 
        label_smoothing=ls
    )

    torch.save(model, output_filepath)
    print(f"Model saved to {output_filepath}.")


def main():

    parser = argparse.ArgumentParser(prog='python -m mycoai.scripts.train',
        description='Trains a deep neural network for taxonomic classification \
            of fungal ITS sequences.')
    
    parser.add_argument('fasta_filepath',
        help='Path to the FASTA file containing ITS sequences for training.')
    
    parser.add_argument('--out', 
        default= [utils.OUTPUT_DIR + 'model.pt'],
        type=str,
        nargs=1,
        help="Path to where the trained model should be saved to (default is \
            model.pt).")
    
    parser.add_argument('--base_arch_type',
        default=['BERT'],
        type=str,
        nargs=1,
        choices=['CNN', 'BERT'],
        help="Type of the to-be-trained base architecture (default is BERT).")
    
    parser.add_argument('--output_type',
        default=['multi'],
        type=str,
        nargs=1,
        choices=['multi', 'infer_sum'],
        help="Whether to use the multi-head output or not (default is multi).")

    parser.add_argument('--no_hls',
        action='store_true',
        help='If specified, turns off hierarchical label smoothing')
    
    parser.add_argument('--levels', 
        nargs='+',
        default=utils.LEVELS,
        help="Specifies the levels that should be trained (or their weights).\
            Can be a list of strings, e.g. ['genus', 'species]. Can also be a \
            list of floats, indicating the weight per level, e.g. \
            [0,0,0,0,1,1] (default is all levels from phylum to species).") 
    
    parser.add_argument('--epochs', 
        default=[50],
        type=int,
        nargs=1,
        help="Number of epochs to train for (default is 50).")
    
    parser.add_argument('--batch_size',
        default=[64],
        type=int,
        nargs=1,
        help="Number of samples per batch (default is 64).")
    
    parser.add_argument('--valid_split',
        default=[0.1],
        type=float,
        nargs=1,
        help="Fraction of data to be used for validation (default is 0.1).")
    
    parser.add_argument('--learning_rate',
        default=[0.0001],
        type=float,
        nargs=1,
        help="Learning rate in Adam optimizer (default is 0.0001).")

    parser.add_argument('--weighted_loss',
        default=[0.5],
        type=float,
        nargs=1,
        help="Strength of how much per-class-loss should be weighted by the \
              reciprocal class size, for imbalanced classes (default is 0.5).")

    parser.add_argument('--device', 
        default=[utils.DEVICE],
        type=str,
        nargs=1,
        help="Forces use of specific device (GPU/CPU). By default, MycoAI will\
              look for and use GPUs whenever available.")

    args = parser.parse_args()
    train(args.fasta_filepath, args.out[0], args.base_arch_type[0], 
          args.output_type[0], not args.no_hls, args.levels, args.epochs[0], 
          args.batch_size[0], args.valid_split[0], args.learning_rate[0], 
          args.weighted_loss[0], args.device[0])
    

if __name__ == '__main__':

    main()