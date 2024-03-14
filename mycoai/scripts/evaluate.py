import argparse
import pandas as pd
from mycoai.evaluate import Evaluator
from mycoai.data import Data
from mycoai import utils

def evaluate(classification, reference, 
             output_filepath=utils.OUTPUT_DIR + 'evaluate.csv'):
    '''Evaluates a predicted classification in comparison to a reference.'''

    classification = pd.read_csv(classification)
    if reference.endswith('.csv'):
        reference = pd.read_csv(reference)
    else:
        reference = Data(reference, allow_duplicates=True)

    evaluator = Evaluator(classification, reference)
    results = evaluator.test()
    evaluator.wandb_finish()
    results.to_csv(output_filepath)
    print(f'Results saved to {output_filepath} and visualized on W&B (link).')
    

def main():

    parser = argparse.ArgumentParser(prog='python -m mycoai.scripts.evaluate',
        description='Evaluates predicted classification of fungal ITS \
            sequences.')
 
    parser.add_argument('classification',
        help='Path to .csv file containing predicted labels.')
    
    parser.add_argument('reference',
        help='Path to .csv or FASTA file containing ground truth labels.')

    parser.add_argument('--out',
        default= [utils.OUTPUT_DIR + 'evaluate.csv'],
        type=str,
        nargs=1,
        help='Where to save the output to (default is evaluate.csv).')
    
    args = parser.parse_args()
    evaluate(args.classification, args.reference)


if __name__ == '__main__':

    main()