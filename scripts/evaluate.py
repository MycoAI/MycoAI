import argparse
import torch
from mycoai import utils
from mycoai.evaluate import Evaluator
from mycoai.data import Data

def evaluate(classification, reference):
    '''TODO'''

    classification = 'TODO' # Read from csv I guess
    reference = Data(reference, allow_duplicates=True) # What if this is a csv?
    evaluator = Evaluator(classification, reference)
    evaluator.test(target_levels) # Add this as argument?
    evaluator.wandb_finish()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python evaluate.py',
        description='Evaluates predicted classification of fungal ITS \
            sequences.')

    # TODO    
    # parser.add_argument('fasta_filepath',
    #     help='Path to the FASTA file containing ITS sequences.')

    # parser.add_argument('--out',
    #     default='prediction.csv',
    #     type=str,
    #     nargs=1,
    #     help='Where to save the output to.')
    
    # parser.add_argument('--method', 
    #     default='deep_its', 
    #     type=str, 
    #     nargs=1, 
    #     choices=['deep_its'],
    #     help="Which classification method to use (default is 'deep_its').")
    
    # parser.add_argument('--load_net',
    #     default='models/test2.pt',
    #     type=set
    #     nargs=1,
    #     help="Path to saved DeepITSClassifier Pytorch model.")
    
    args = parser.parse_args()
    evaluate()