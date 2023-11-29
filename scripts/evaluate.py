import argparse
import pandas as pd
from mycoai.evaluate import Evaluator
from mycoai.data import Data

def evaluate(classification, reference):
    '''Evaluates a predicted classification in comparison to a reference.'''

    classification = pd.read_csv(classification)
    if reference.endswith('.csv'):
        reference = pd.read_csv(reference)
    else:
        reference = Data(reference, allow_duplicates=True)

    evaluator = Evaluator(classification, reference)
    evaluator.test()
    evaluator.wandb_finish()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python evaluate.py',
        description='Evaluates predicted classification of fungal ITS \
            sequences.')
 
    parser.add_argument('classification',
        help='Path to .csv file containing predicted labels.')
    
    parser.add_argument('reference',
        help='Path to .csv or FASTA file containing ground truth labels.')
    
    args = parser.parse_args()
    evaluate(args.classification, args.reference)