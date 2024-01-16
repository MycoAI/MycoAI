from train import Train
from classify import Classify
import argparse
import sys
from pathlib import Path

script_directory = Path(__file__).parent.absolute()
project_directory = script_directory.parent.absolute()

sys.path.append(str(project_directory))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python its_classifier.py',
        description='Taxonomic classification of fungal ITS sequences.', usage="%(prog)s <positional arguments> [options]")

    '''Predicts the taxonomies of sequences in file with specified method'''

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for the dnabarcoder predict comman
    train_dnabarcoder_parser = subparsers.add_parser('train_dnabarcoder', help='Train a dnabarcoder by computing cutoffs for taxonomic classification of fungal ITS sequences.')




    # Subparser for the "deep_its" command
    train_deep_parser = subparsers.add_parser('train_deep', help='Train a deep learning model for taxonomic classification of fungal ITS sequences.')

    # Subparser for the "its" command
    classify_dnabarcoder_parser = subparsers.add_parser('classify_dnabarcoder', help='Classify fungal ITS sequences using dnabarcoder')


    # Subparser for the "its" command
    classify_deep_parser = subparsers.add_parser('classify_deep', help='Classify fungal ITS sequences using deep learning')


    trainer = Train(train_dnabarcoder_parser, train_deep_parser)
    trainer.add_dnabarcoder_args()
    trainer.add_deep_args()
    classifier = Classify(classify_dnabarcoder_parser, classify_deep_parser)
    classifier.add_blast_args()
    classifier.add_deep_args()

    args = parser.parse_args()

    if args.command == 'train_deep':
        trainer.deep(args)
    elif args.command == 'train_dnabarcoder':
        trainer.dnabarcoder(args)
    elif args.command == 'classify_deep':
        classifier.deep(args)
