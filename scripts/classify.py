import argparse
import sys
import os
import torch
from pathlib import Path

script_directory = Path(__file__).parent.absolute()
parent_directory = script_directory.parent.absolute()

sys.path.append(str(parent_directory))


from mycoai import utils, data
from mycoai.loggingwrapper import LoggingWrapper
from mycoai.models import BLASTClassifier


def add_blast_args(group):
    group.add_argument('--method', choices=['blast'], required=True)
    group.add_argument('--fasta_filepath',
                            help='Path to the FASTA file containing ITS sequences.')

    group.add_argument('--out',
                            default='prediction.csv',
                            type=str,
                            nargs=1,
                            help='Where to save the output to.')
    group.add_argument('-r', '--reference', required=True, help='the reference fasta file.')
    group.add_argument('-t', '--threshold', required=True, type=float, default=0.97,
                             help='The threshold for the classification.')
    group.add_argument('-mc', '--mincoverage', type=int, default=300,
                             help='Optinal. Minimum coverage required for the identitiy of the BLAST comparison.')
    group.add_argument('-c', '--classification', help='the classification file in tab. format.')  # optinal
    group.add_argument('-p', '--classificationpos', type=int, default=0,
                             help='the classification position to load the classification.')  # optional

def add_its_args(group):
    group.add_argument('-m', '--method', choices=['simpleCnn','dbn', 'cnn', 'rdp', 'transformer'], required=True)
    group.add_argument('--fasta_filepath',
                            help='Path to the FASTA file classify containing ITS sequences.')

    group.add_argument('--out',
                            default='prediction.csv',
                            type=str,
                            nargs=1,
                            help='Where to save the output to.')


def add_classify_args(parser):

    blast_arg_group = parser.add_argument_group('blast', 'Arguments for BLAST classifier.')
    deep_its_arg_group = parser.add_argument_group('deep_its', 'Arguments for deep ITS classifier.')
    add_blast_args(blast_arg_group)
    add_its_args(deep_its_arg_group)



def classify(args):


    if args.command=='deep_its':
        deep_its_model = torch.load('models/test2.pt') # replace with model
        deep_its_model.to(utils.device),
        prediction = deep_its_model.classify(args.fasta_filepath)
        prediction.to_csv(utils.output_dir + args.out)
    elif args.command  =='blast':
        loggingwrapper.info("using blast classifier...")
        blastclassifier = blastclassifier(args)
        blastclassifier.classify()

