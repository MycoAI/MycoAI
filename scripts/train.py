import ast
import sys
from pathlib import Path

script_directory = Path(__file__).parent.absolute()
parent_directory = script_directory.parent.absolute()

sys.path.append(str(parent_directory))


from mycoai import utils, data
from mycoai.loggingwrapper import LoggingWrapper
from mycoai.models import BLASTClassifier
import torch
from mycoai import data, utils, plotter
from mycoai.models import ITSClassifier
from mycoai.models.architectures import ResNet, CNN
from mycoai.training import ClassificationTask

import configparser

class TrainConfig:
    def __init__(self, file_path=None):
        # Default values
        self.dna_encoder = 'kmer-spectral'
        self.tax_encoder = 'categorical'
        self.epochs = 100
        self.batch_size = 32
        self.fcn_layers = [128, 20, 64]
        self.optimizer = 'Adam'
        self.sampler = 'random'
        self.loss_function = 'CrossEntropyLoss'
        self.output_heads= 'single'
        self.weight_schedule = 'Constant([1,1,1,1,1,1])'
        self.warmup_steps = None
        self.dropout = 0.0
        self.sequence_length_filter_tolerance = None
        self.sequence_quality_filter_tolerance = None


        if file_path is not None:
            self._load_from_file(file_path)

    def _load_from_file(self, file_path):
        config = configparser.ConfigParser()
        config.read(file_path)

        # Load DEFAULT section
        for key, value in config['DEFAULT'].items():
            setattr(self, key, self._convert_type(value))

        # Load HYPERPARAMETER section, overriding DEFAULT values
        if 'HYPERPARAMETER' in config:
            for key, value in config['HYPERPARAMETER'].items():
                setattr(self, key, self._convert_type(value))

    def _convert_type(self, value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value



    def set_epochs(self, epochs):
        self.epochs = epochs
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_fcn_layers(self, fcn_layers):
        self.fcn_layers = fcn_layers
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    def set_weight_schedule(self, weight_schedule):
        self.weight_schedule = weight_schedule
    def set_warmup_steps(self, warmup_steps):
        self.warmup_steps = warmup_steps
    def set_dropout(self, dropout):
        self.dropout = dropout

    def set_sampler(self, sampler):
        self.sampler = sampler
    def set_loss_function(self, loss_function):
        self.loss_function = loss_function
    def set_output_heads(self, output_heads):
        self.output_heads = output_heads
    def set_dna_encoder(self, dna_encoder):
        self.dna_encoder = dna_encoder
    def set_tax_encoder(self, tax_encoder):
        self.tax_encoder = tax_encoder
    def set_sequence_length_filter_tolerance(self, sequence_length_filter_tolerance):
        self.sequence_length_filter_tolerance = sequence_length_filter_tolerance
    def set_sequence_quality_filter_tolerance(self, sequence_quality_filter_tolerance):
        self.sequence_quality_filter_tolerance = sequence_quality_filter_tolerance



class Train:

    def __init__(self, train_blast_parser, train_deep_parser):
        self.train_blast_parser = train_blast_parser
        self.train_deep_parser = train_deep_parser

    def add_train_blast_args(self):
        self.train_blast_parser.add_argument('--fasta_filepath',
                                help='Path to the FASTA file containing ITS sequences.')

        self.train_blast_parser.add_argument('--out',
                                default='prediction.csv',
                                type=str,
                                nargs=1,
                                help='Where to save the output to.')
        self.train_blast_parser.add_argument('-r', '--reference', required=True, help='the reference fasta file.')
        self.train_blast_parser.add_argument('-t', '--threshold', required=True, type=float, default=0.97,
                                 help='The threshold for the classification.')
        self.train_blast_parser.add_argument('-mc', '--mincoverage', type=int, default=300,
                                 help='Optinal. Minimum coverage required for the identitiy of the BLAST comparison.')
        self.train_blast_parser.add_argument('-c', '--classification', help='the classification file in tab. format.')  # optinal
        self.train_blast_parser.add_argument('-p', '--classificationpos', type=int, default=0,
                                 help='the classification position to load the classification.')  # optional



    def add_train_deep_args(self):
        self.train_deep_parser.add_argument('--model', type=str, choices=['resnet', 'cnn', 'simpleCnn', 'dbn', 'rdp', 'transformer'], help='Neural network', default='cnn')
        self.train_deep_parser.add_argument('--outdir', type=str, help='Directory to save the output', required=True)
        self.train_deep_parser.add_argument('--train_data', type=str, help='FASTA file for training', required=True)
        self.train_deep_parser.add_argument('--valid_data_split', type=float, choices=range(0, 1),
                            help='Fraction of training data to use for validation. Default is 0.2', default=0.2)
        self.train_deep_parser.add_argument('--target_levels', type=list,
                            help='Taxonomic levels to predict. Default is all levels')
        self.train_deep_parser.add_argument('--metrics', type=str,
                            help='Evaluation metrics to report during training, provided as dictionary with metric name as key and function as value. Default is accuracy, balanced acuracy, precision, recall, f1, and mcc.',
                            default='{"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}')
        self.train_deep_parser.add_argument('--wandb_config', type=str,
                            help='Allows the user to add extra information to the weights and biases config data. Default is {}',
                            default='{}')
        self.train_deep_parser.add_argument('--wandb_name', type=str,
                            help='Name of the run to be displayed on weights and biases. Will choose a random name if unspecified. Default is None',
                            default=None)
        self.train_deep_parser.add_argument('--cuda', type=int, help='Use CUDA enabled GPU if available. The number indicates the GPU to use', default=None)
        self.train_deep_parser.add_argument('--config', type=str, help='Path to config file. Default is None', required=True, default=None)


    def train_deep(self, args):

        hyperparameters = TrainConfig(args.config)
        # Cuda setup
        if args.cuda is not None:
            utils.set_device('cuda:' + str(args.cuda))  # To specify GPU use

        # Data import & preprocessing
        train_data = data.DataPrep(args.train_data)
        #train_data = train_data.class_filter('species', min_samples=5)
        if hyperparameters.sequence_length_filter_tolerance is not None:
            train_data = train_data.sequence_length_filter(tolerance=hyperparameters.sequence_length_filter_tolerance)
        if hyperparameters.sequence_quality_filter_tolerance is not None:
            train_data = train_data.sequence_quality_filter(tolerance=hyperparameters.sequence_quality_filter_tolerance)
        train_data, valid_data = train_data.encode_dataset(hyperparameters.dna_encoder, hyperparameters.tax_encoder, valid_split=args.valid_data_split)


        # Model definition
        if (args.model == 'resnet'):
            arch = ResNet([2, 2, 2, 2])
        elif (args.model == 'cnn'):
            arch = CNN(len(utils.LEVELS))
        else:
            raise ValueError("Model not supported")
        model = ITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder, hyperparameters.fcn_layers, hyperparameters.output_heads, args.target_levels, hyperparameters.dropout)

        model, history = ClassificationTask.train(model, train_data, valid_data, 100)
        plotter.classification_loss(history, model.target_levels)
        torch.save(model, args.outdir/args.model + '_model.pt')

