import argparse
import torch
from mycoai import utils, data

def classify(fasta_filepath, output_filepath='prediction.csv', 
             method='deep_its'):
    '''Predicts the taxonomies of sequences in file with specified method'''
    
    if method=='deep_its':
        deep_its_model = torch.load('models/test2.pt') # Replace with model
        deep_its_model.to(utils.DEVICE)
        prediction = deep_its_model.classify(fasta_filepath)
        prediction.to_csv(utils.OUTPUT_DIR + output_filepath)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python classify.py',
        description='Taxonomic classification of fungal ITS sequences.',
        epilog='test')
    
    parser.add_argument('fasta_filepath',
        help='Path to the FASTA file containing ITS sequences.')

    parser.add_argument('--out',
        default='prediction.csv',
        type=str,
        nargs=1,
        help='Where to save the output to.')
    
    parser.add_argument('--method', 
        default='deep_its', 
        type=str, 
        nargs=1, 
        choices=['deep_its'],
        help="Which classification method to use (default is 'deep_its').")
    
    args = parser.parse_args()
    classify(args.fasta_filepath, method=args.method)