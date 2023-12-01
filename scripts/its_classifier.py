from train import Train
from classify import Classify
import argparse
import sys
from pathlib import Path

from  mycoai.loggingwrapper import LoggingWrapper

script_directory = Path(__file__).parent.absolute()
parent_directory = script_directory.parent.absolute()

sys.path.append(str(parent_directory))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python its_classifier.py',
        description='Taxonomic classification of fungal ITS sequences.', usage="%(prog)s <positional arguments> [options]")

    '''Predicts the taxonomies of sequences in file with specified method'''

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for the "blast" command
    train_blast_parser = subparsers.add_parser('train_blast')

    # Subparser for the "deep_its" command
    train_deep_parser = subparsers.add_parser('train_deep')

    # Subparser for the "its" command
    classify_blast_parser = subparsers.add_parser('classify_blast')


    # Subparser for the "its" command
    classify_deep_parser = subparsers.add_parser('classify_deep')


    trainer = Train(train_blast_parser, train_deep_parser)
    trainer.add_blast_args()
    trainer.add_deep_args()
    classifier = Classify(classify_blast_parser, classify_deep_parser)
    classifier.add_blast_args()
    classifier.add_deep_args()

    args = parser.parse_args()

    if args.command == 'train_deep':
        trainer.deep(args)
    elif args.command == 'classify_deep':
        classifier.deep(args)
    else:
        print('Invalid command. Please try again.')