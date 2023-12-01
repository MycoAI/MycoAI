import ast
import sys
from pathlib import Path


script_directory = Path(__file__).parent.absolute()
parent_directory = script_directory.parent.absolute()

sys.path.append(str(parent_directory))

from mycoai.deep.train import DeepITSTrainer

from mycoai import utils, data
from mycoai.loggingwrapper import LoggingWrapper
from mycoai.trad import BLASTClassifier
import torch
from  mycoai.data import Data
from mycoai import utils, plotter
from mycoai.deep.models import deep_its_classifier, BERT, DeepITSClassifier
from mycoai.deep.models.architectures import ResNet, CNN
from mycoai.deep.train.weight_schedules import Constant
from mycoai.deep.train.weight_schedules import Constant

#from mycoai.training import ClassificationTask

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
        self.sample = True
        self.loss_function = 'CrossEntropyLoss'
        self.output_heads= 'single'
        self.weighted_loss = True
        self.warmup_steps = None
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.dropout = 0.0
        self.sequence_length_filter_tolerance = None
        self.sequence_quality_filter_tolerance = None
        self.weight_schedule = [1,1,1,1,1,1]


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
    def set_weighted_loss(self, weighted_loss):
        self.weighted_loss = weighted_loss
    def set_warmup_steps(self, warmup_steps):
        self.warmup_steps = warmup_steps
    def set_dropout(self, dropout):
        self.dropout = dropout

    def set_sample(self, sample):
        self.sampler = sample
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

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
    def set_weight_schedule(self, weight_schedule):
        self.weight_schedule = weight_schedule

class Train:

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
        self.blast_parser.add_argument('-c', '--classification', help='the classification file in tab. format.')  # optinal
        self.blast_parser.add_argument('-p', '--classificationpos', type=int, default=0,
                                 help='the classification position to load the classification.')  # optional



    def add_deep_args(self):
        self.deep_parser.add_argument('--model', type=str, choices=['resnet', 'cnn', 'simpleCnn', 'dbn', 'rdp', 'bert'], help='Neural network', default='cnn')
        self.deep_parser.add_argument('--model_path', type=str, help='Path to save the model', default='model.pt')
        self.deep_parser.add_argument('--train_data', type=str, help='FASTA file for training', required=True)
        self.deep_parser.add_argument('--valid_data_split', type=float,
                            help='Fraction of training data to use for validation. Default is 0.2', default=0.2)
        self.deep_parser.add_argument('--target_levels', type=list,
                            help='Taxonomic levels to predict. Default is all levels' + str(utils.LEVELS), default=utils.LEVELS)
        self.deep_parser.add_argument('--metrics', type=dict,
                           help='Evaluation metrics to report during training, provided as dictionary with metric name as key and function as value. Default is accuracy, balanced acuracy, precision, recall, f1, and mcc.',
                          default=utils.EVAL_METRICS)
        self.deep_parser.add_argument('--wandb_config', type=dict,
                            help='Allows the user to add extra information to the weights and biases config data. Default is {}',
                            default={})
        self.deep_parser.add_argument('--wandb_name', type=str,
                            help='Name of the run to be displayed on weights and biases. Will choose a random name if unspecified. Default is None',
                            default=None)
        self.deep_parser.add_argument('--gpu', type=int, const=0, nargs='?', help='Use CUDA enabled GPU if available. The number indicates the GPU to use', default=None)
        self.deep_parser.add_argument('--config', type=str, help='Path to config file. Default is None', default=None)


    def deep(self, args):

        hyperparameters = TrainConfig(args.config)
        # Cuda setup
        if args.gpu is not None:
            utils.set_device('cuda:' + str(args.cuda))  # To specify GPU use

        if args.model == 'cnn':
            hyperparameters.fcn_layers= []
            hyperparameters.output_heads = 'single'
            hyperparameters.dna_encoder = 'kmer-spectral'
        elif args.model == 'resnet':
            hyperparameters.dna_encoder = '4d'
            hyperparameters.output_heads = 'multi'
        elif args.model == 'bert':
            hyperparameters.fcn_layers = []
            hyperparameters.output_heads = 'infer_parent'
            hyperparameters.dna_encoder = 'bpe'

        # Data import & preprocessing
        in_data = Data(args.train_data)
        if args.valid_data_split > 0:
            train_data, valid_data =  train_data, valid_data = in_data.encode_dataset(hyperparameters.dna_encoder, hyperparameters.tax_encoder, valid_split=args.valid_data_split)
        else:
            train_data, valid_data = in_data.encode_dataset(hyperparameters.dna_encoder, hyperparameters.tax_encoder, valid_split=args.valid_data_split)
            valid_data = None

        if hyperparameters.sequence_length_filter_tolerance is not None:
            train_data = train_data.sequence_length_filter(tolerance=hyperparameters.sequence_length_filter_tolerance)
        if hyperparameters.sequence_quality_filter_tolerance is not None:
            train_data = train_data.sequence_quality_filter(tolerance=hyperparameters.sequence_quality_filter_tolerance)



        # Model definition
        if (args.model == 'resnet'):
            base_arch = ResNet([2, 2, 2, 2])
        elif (args.model == 'cnn'):
            base_arch = CNN(len(utils.LEVELS))
        elif (args.model == 'bert'):
            base_arch = BERT(train_data.dna_encoder.len_input,
                             train_data.dna_encoder.vocab_size)

        else:
            raise ValueError("Model not supported")
        model = DeepITSClassifier(base_arch=base_arch, dna_encoder=train_data.dna_encoder, tax_encoder=train_data.tax_encoder, target_levels=args.target_levels, fcn_layers=hyperparameters.fcn_layers, output=hyperparameters.output_heads, dropout=hyperparameters.dropout)
        # Training
        sampler = train_data.weighted_sampler() if hyperparameters.sample else None
        loss = None
        if hyperparameters.weighted_loss is not None:
            loss = train_data.weighted_loss(torch.nn.CrossEntropyLoss, sampler)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate, weight_decay=hyperparameters.weight_decay)
        model, history = DeepITSTrainer.train(model=model, train_data=train_data, valid_data=valid_data, epochs=hyperparameters.epochs, loss=loss, batch_size=hyperparameters.batch_size, sampler=sampler, optimizer=optimizer, metrics=args.metrics, wandb_config=args.wandb_config, wandb_name=args.wandb_name, weight_schedule=Constant(hyperparameters.weight_schedule), warmup_steps=hyperparameters.warmup_steps)
        torch.save(model, args.model_path)
