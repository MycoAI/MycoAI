import argparse
import torch
from mycoai import utils
from mycoai.data import Data
from mycoai.deep.models import DeepITSClassifier, ResNet, BERT
from mycoai.deep.train import DeepITSTrainer
from mycoai.deep.train.weight_schedules import Constant

def train(fasta_filepath, save_model, base_arch_type, epochs, batch_size, 
          valid_split, learning_rate, weight_decay, weighted_loss, 
          weighted_sampling, device):
    '''Trains a Deep ITS classifier.'''

    utils.set_device(device)

    # Inferring dna_encoder from base arch
    if base_arch_type == 'BERT':
        dna_encoder = 'bpe'
    if base_arch_type == 'ResNet':
        dna_encoder = '4d'

    # Reading/encoding the data
    data = Data(fasta_filepath)
    if valid_split > 0:
        train_data, valid_data = data.encode_dataset(dna_encoder, 
                                                     valid_split=valid_split)
    else:
        train_data = data.encode_dataset(dna_encoder, valid_split=valid_split)
        valid_data = None

    # Defining model
    if base_arch_type == 'BERT':
        base_arch = BERT(train_data.dna_encoder.length, 
                         train_data.dna_encoder.vocab_size)
        lvl_weights = Constant([0,0,0,0,0,1])
        output_head = 'infer_parent'
        fcn = []
    else:
        base_arch = ResNet([2,2,2,2])
        lvl_weights = Constant([1,1,1,1,1,1])
        output_head = 'multi'
        fcn = [128,50,64]
    model = DeepITSClassifier(base_arch, train_data.dna_encoder, 
                              train_data.tax_encoder, fcn, output_head)
    
    # Training
    sampler = train_data.weighted_sampler() if weighted_sampling else None
    loss = None
    if weighted_loss:
        loss = train_data.weighted_loss(torch.nn.CrossEntropyLoss, sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay)
    model, history = DeepITSTrainer.train(model, train_data, valid_data, epochs,
                                          loss, batch_size, sampler, optimizer,
                                          weight_schedule=lvl_weights)
    torch.save(model, save_model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python train.py',
        description='Trains a deep neural network for taxonomic classification \
            of fungal ITS sequences.')
    
    parser.add_argument('fasta_filepath',
        help='Path to the FASTA file containing ITS sequences for training.')
    
    parser.add_argument('--save_model', 
        default= [utils.OUTPUT_DIR + 'model.pt'],
        type=str,
        nargs=1,
        help="Path to where the trained model should be saved to.")
    
    parser.add_argument('--base_arch_type',
        default=['BERT'],
        type=str,
        nargs=1,
        choices=['ResNet', 'BERT'],
        help="Type of the to-be-trained base architecture (default is BERT).")
    
    parser.add_argument('--epochs', 
        default=[100],
        type=int,
        nargs=1,
        help="Number of epochs to train for (default is 100).")
    
    parser.add_argument('--batch_size',
        default=[64],
        type=int,
        nargs=1,
        help="Number of samples per batch (default is 64).")
    
    parser.add_argument('--valid_split',
        default=[0.2],
        type=float,
        nargs=1,
        help="Fraction of data to be used for validation (default is 0.2).")
    
    parser.add_argument('--learning_rate',
        default=[0.0001],
        type=float,
        nargs=1,
        help="Learning rate in Adam optimizer (default is 0.0001).")
    
    parser.add_argument('--weight_decay',
        default=[0],
        type=float,
        nargs=1,
        help="Weight decay in Adam optimizer (regularization) (default is 0).")

    parser.add_argument('--weighted_loss',
        default=[1],
        type=int,
        nargs=1,
        choices=[0,1],
        help="Whether to weigh the loss by the reciprocal class size \
              (default is 1).")
    
    parser.add_argument('--weighted_sampling',
        default=[0],
        type=int,
        nargs=1,
        choices=[0,1],
        help="Whether to sample data with a probability that is based on the \
              reciprocal of their class size (default is 0).")

    parser.add_argument('--device', 
        default=[utils.DEVICE],
        type=str,
        nargs=1,
        help="Forces use of specific device (GPU/CPU). By default, MycoAI will\
              look for and use GPUs whenever available.")

    args = parser.parse_args()
    train(args.fasta_filepath, args.save_model[0], args.base_arch_type[0], 
          args.epochs[0], args.batch_size[0], args.valid_split[0], 
          args.learning_rate[0], args.weight_decay[0], 
          bool(args.weighted_loss[0]), bool(args.weighted_sampling[0]), 
          args.device[0])