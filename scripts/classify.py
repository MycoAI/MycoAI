import argparse
import sys

import torch
from mycoai import utils, data
from mycoai.loggingwrapper import LoggingWrapper
from mycoai.models import BLASTClassifier

def classify(fasta_filepath, output_filepath='prediction.csv', prog_args=None):
    '''Predicts the taxonomies of sequences in file with specified method'''

    if args.command=='deep_its':
        deep_its_model = torch.load('models/test2.pt') # Replace with model
        deep_its_model.to(utils.DEVICE)
        prediction = deep_its_model.classify(fasta_filepath)
        prediction.to_csv(utils.OUTPUT_DIR + output_filepath)
    elif args.command  =='blast':
        LoggingWrapper.info("Using BLAST classifier...")
        blastClassifier = BLASTClassifier(prog_args, fasta_filepath, output_filepath)
        blastClassifier.classify()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python classify.py',
        description='Taxonomic classification of fungal ITS sequences.')

    parser.add_argument('--fasta_filepath',
        help='Path to the FASTA file containing ITS sequences.')

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for the "blast" command
    blast_parser = subparsers.add_parser('blast')

    # Subparser for the "its" command
    its_parser = subparsers.add_parser('deep_its')

    parser.add_argument('--out',
        default='prediction.csv',
        type=str,
        nargs=1,
        help='Where to save the output to.')


    args = parser.parse_args()
    classify(args.fasta_filepath, args.out, args)