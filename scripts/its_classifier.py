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

    # Subparser for the dnabarcoder predict comman
    train_dnabarcoder_parser = subparsers.add_parser('dnabarcoder_train')




    # Subparser for the "deep_its" command
    train_deep_parser = subparsers.add_parser('deep_train')

    # Subparser for the "its" command
    classify_dnabarcoder_parser = subparsers.add_parser('dnabarcoder_classify')


    # Subparser for the "its" command
    classify_deep_parser = subparsers.add_parser('deep_classify')


    trainer = Train(train_dnabarcoder_parser, train_deep_parser)
    trainer.add_dnabarcoder_args()
    trainer.add_deep_args()
    classifier = Classify(classify_dnabarcoder_parser, classify_deep_parser)
    classifier.add_blast_args()
    classifier.add_deep_args()

    args = parser.parse_args()

    if args.command == 'train_deep':
        trainer.deep(args)
    elif args.command == 'classify_deep':
        classifier.deep(args)
    else:
        print('Invalid command. Please try again.')