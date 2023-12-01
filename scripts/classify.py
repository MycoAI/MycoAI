# Description: This script is used to classify ITS sequences using BLAST and DeepITS.

import sys
from pathlib import Path

script_directory = Path(__file__).parent.absolute()
parent_directory = script_directory.parent.absolute()

sys.path.append(str(parent_directory))

from mycoai.loggingwrapper import LoggingWrapper
from mycoai.trad import BLASTClassifier
import torch
from mycoai import utils, plotter



class Classify:

    def __init__(self, blast_parser, deep_parser):
        self.blast_parser = blast_parser
        self.deep_parser = deep_parser

    def add_blast_args(self):
        self.blast_parser.add_argument('--fasta_filepath',
                                             help='Path to the FASTA file containing ITS sequences.')

        self.blast_parser.add_argument('--out',
                                             default='prediction.csv',
                                             type=str,
                                             nargs=1,
                                             help='Where to save the output to.')
        self.blast_parser.add_argument('-r', '--reference', required=True, help='the reference fasta file.')
        self.blast_parser.add_argument('-t', '--threshold', required=True, type=float, default=0.97,
                                             help='The threshold for the classification.')
        self.blast_parser.add_argument('-mc', '--mincoverage', type=int, default=300,
                                             help='Optinal. Minimum coverage required for the identitiy of the BLAST comparison.')
        self.blast_parser.add_argument('-c', '--classification',
                                             help='the classification file in tab. format.')  # optinal
        self.blast_parser.add_argument('-p', '--classificationpos', type=int, default=0,
                                             help='the classification position to load the classification.')  # optional

    def add_deep_args(self):
        self.deep_parser.add_argument('--load_model', type=str, help='Path to save the model', required=True)

        self.deep_parser.add_argument('--fasta_filepath',
                           help='Path to the FASTA file classify containing ITS sequences.', required=True)

        self.deep_parser.add_argument('--out',
                           default='prediction.csv',
                           type=str,
                           nargs=1,
                           help='Path to the output CSV file.')
        self.deep_parser.add_argument('--gpu', type=int, const=0, nargs='?',
                                      help='Use CUDA enabled GPU if available. The number indicates the GPU to use',
                                      default=None)

    def deep(self, args):
        deep_its_model = torch.load(args.load_model)
        if args.gpu is not None:
            utils.set_device('cuda:' + str(args.cuda))  # To specify GPU use
        prediction = deep_its_model.classify(args.fasta_filepath)
        prediction.to_csv(args.out)
    def blast(self, args):
        blastclassifier = BLASTClassifier(args)
        blastclassifier.classify()