import argparse
import torch
from mycoai import utils

def classify(fasta_filepath, output_filepath='prediction.csv', 
             model='models/test2.pt'):
    '''Predicts the taxonomies of sequences in file with specified method'''
    
    deep_its_model = torch.load(model)
    deep_its_model.to(utils.DEVICE)
    prediction = deep_its_model.classify(fasta_filepath)
    prediction.to_csv(utils.OUTPUT_DIR + output_filepath) # TODO

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python classify.py',
        description='Taxonomic classification of fungal ITS sequences using a\
            deep neural network.')
    
    parser.add_argument('fasta_filepath',
        help='Path to the FASTA file containing ITS sequences.')

    parser.add_argument('--out', '-o',
        default='prediction.csv',
        type=str,
        nargs=1,
        help='Where to save the output to.')
    
    parser.add_argument('--load_model', '-m', 
        default='models/test2.pt',
        type=str,
        nargs=1,
        help="Path to saved DeepITSClassifier Pytorch model.")
    
    args = parser.parse_args()
    classify(args.fasta_filepath, output_filepath=args.out, 
             model=args.load_model)