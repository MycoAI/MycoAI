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

#from mycoai.training import ClassificationTask

import configparser

class TrainConfig:
    ''' Class to store hyperparameters for training. Can be loaded from a config file.'''
    def __init__(self):
        # Default values
        self.dna_encoder = 'kmer-spectral'
        self.kmer_size = 4
        self.tax_encoder = 'categorical'
        self.epochs = 10
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
        self._load_from_file("hyperparameters.ini")

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



class Train:

    def __init__(self, dnabarcoder_parser, deep_parser):
        self.dnabarcoder_parser = dnabarcoder_parser
        self.deep_parser = deep_parser

    def add_dnabarcoder_args(self):
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
        self.deep_parser.add_argument('--base_arch_type', type=str, choices=['ResNet', 'BERT', 'CNN'], help='Type of the to-be-trained base architecture', default='cnn')
        self.deep_parser.add_argument('--save_model', type=str, help='Path to where the trained model should be saved', default='model.pt')
        self.deep_parser.add_argument('train_data', type=str, help='Path to the FASTA file containing ITS sequences for training.')
        self.deep_parser.add_argument('--validation_split', type=float,
                            help='Fraction of training data to use for validation. Default is 0.2', default=0.2)
        self.deep_parser.add_argument('--target_levels', type=list,
                            help='Taxonomic levels to predict. Default is all levels:' + str(utils.LEVELS), default=utils.LEVELS)
        self.deep_parser.add_argument('--metrics', type=dict,
                           help='Evaluation metrics to report during training, provided as dictionary with metric name as key and function as value. Default is accuracy, balanced acuracy, precision, recall, f1, and mcc.',
                          default=utils.EVAL_METRICS)
        self.deep_parser.add_argument('--wandb_config', type=dict,
                            help='Allows the user to add extra information to the weights and biases config data. Default is {}',
                            default={})
        self.deep_parser.add_argument('--wandb_name', type=str,
                            help='Name of the run to be displayed on weights and biases. Will choose a random name if unspecified. Default is None',
                            default=None)
        self.deep_parser.add_argument('--gpu', type=int, const=0, nargs='?', help='Use CUDA enabled GPU if available. The number following the argument indicates the GPU to use in a multi-GPU system', default=None)
        self.deep_parser.add_argument('--hyperparameters', type=str, help='Path to hyperparameters config file. Default is None', default=None)


    def deep(self, args):

        hyperparameters = TrainConfig()
        # Cuda setup
        if args.gpu is not None:
            utils.set_device('cuda:' + str(args.cuda))  # To specify GPU use
        else:
            utils.set_device('cpu')
        if args.base_arch_type == 'CNN':
            hyperparameters.fcn_layers= []
            hyperparameters.output_heads = 'multi'
            hyperparameters.dna_encoder = 'kmer-spectral'
            hyperparameters.kmer_size = 6
        elif args.base_arch_type == 'ResNet':
            hyperparameters.dna_encoder = '4d'
            hyperparameters.output_heads = 'multi'
        elif args.base_arch_type == 'BERT':
            hyperparameters.fcn_layers = []
            hyperparameters.output_heads = 'infer_parent'
            hyperparameters.dna_encoder = 'bpe'

        # Data import & preprocessing
        in_data = Data(args.train_data)
        if hyperparameters.sequence_length_filter_tolerance is not None:
            in_data = in_data.sequence_length_filter(tolerance=hyperparameters.sequence_length_filter_tolerance)
        if hyperparameters.sequence_quality_filter_tolerance is not None:
            in_data = in_data.sequence_quality_filter(tolerance=hyperparameters.sequence_quality_filter_tolerance)

        if args.validation_split > 0:
            train_data, valid_data = in_data.encode_dataset(hyperparameters.dna_encoder, hyperparameters.tax_encoder, valid_split=args.validation_split, k=hyperparameters.kmer_size)
        else:
            train_data = in_data.encode_dataset(hyperparameters.dna_encoder, hyperparameters.tax_encoder,valid_split=0.0, k=hyperparameters.kmer_size)
            valid_data = None





        # Model definition
        if (args.base_arch_type == 'ResNet'):
            base_arch = ResNet([2, 2, 2, 2])
        elif (args.base_arch_type == 'CNN'):
            base_arch = CNN(nb_classes=len(utils.LEVELS), input_length=train_data.sequences.shape[2])
        elif (args.base_arch_type == 'BERT'):
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
        torch.save(model, args.save_model)
