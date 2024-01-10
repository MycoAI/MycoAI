import ast
import os
import subprocess
import sys
from pathlib import Path


script_directory = Path(__file__).parent.absolute()
project_directory = script_directory.parent.absolute()

sys.path.append(str(project_directory))

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
        self.dnabarcoder_parser.add_argument('-i', '--input', required=True, help='the fasta file')
        self.dnabarcoder_parser.add_argument('-o', '--out', default="dnabarcoder", help='The output folder.')
        self.dnabarcoder_parser.add_argument('-prefix', '--prefix', default="", help='the prefix of output filenames.')
        self.dnabarcoder_parser.add_argument('-label', '--label', default="", help='The label to display in the figure.')
        self.dnabarcoder_parser.add_argument('-labelstyle', '--labelstyle', default='normal',
                            help='The label style to be displayed: normal, italic, or bold.')
        self.dnabarcoder_parser.add_argument('-c', '--classification', default="", help='the classification file in tab. format.')
        # parser.add_argument('-p','--classificationpositions', default="", help='the classification positions for the prediction, separated by ",".')
        self.dnabarcoder_parser.add_argument('-rank', '--classificationranks', default="",
                            help='the classification ranks for the prediction, separated by ",".')
        self.dnabarcoder_parser.add_argument('-st', '--startingthreshold', type=float, default=0, help='starting threshold')
        self.dnabarcoder_parser.add_argument('-et', '--endthreshold', type=float, default=0, help='ending threshold')
        self.dnabarcoder_parser.add_argument('-s', '--step', type=float, default=0.001,
                            help='the step to be increased for the threshold after each step of the prediction.')
        self.dnabarcoder_parser.add_argument('-ml', '--minalignmentlength', type=int, default=400,
                            help='Minimum sequence alignment length required for BLAST. For short barcode sequences like ITS2 (ITS1) sequences, minalignmentlength should probably be set to smaller, 50 for instance.')
        self.dnabarcoder_parser.add_argument('-sim', '--simfilename', help='The similarity matrix of the sequences if exists.')
        # parser.add_argument('-hp','--higherclassificationpositions', default="", help='The prediction is based on the whole dataset if hp="". Otherwise it will be predicted based on different datasets obtained at the higher classifications, separated by ",".')
        self.dnabarcoder_parser.add_argument('-higherrank', '--higherclassificationranks', default="",
                            help='The prediction is done on the whole dataset if higherranks="". Otherwise it will be predicted for different datasets obtained at the higher classifications, separated by ",".')
        self.dnabarcoder_parser.add_argument('-mingroupno', '--mingroupno', type=int, default=10,
                            help='The minimum number of groups needed for prediction.')
        self.dnabarcoder_parser.add_argument('-minseqno', '--minseqno', type=int, default=30,
                            help='The minimum number of sequences needed for prediction.')
        self.dnabarcoder_parser.add_argument('-maxseqno', '--maxseqno', type=int, default=20000,
                            help='Maximum number of the sequences of the predicted taxon name from the classification file will be selected for the comparison to find the best match. If it is not given, all the sequences will be selected.')
        self.dnabarcoder_parser.add_argument('-maxproportion', '--maxproportion', type=float, default=1,
                            help='Only predict when the proportion of the sequences the largest group of the dataset is less than maxproportion. This is to avoid the problem of inaccurate prediction due to imbalanced data.')
        self.dnabarcoder_parser.add_argument('-taxa', '--taxa', default="",
                            help='The selected taxa separated by commas for local prediction. If taxa=="", all the clades at the given higher positions are selected for prediction.')
        self.dnabarcoder_parser.add_argument('-removecomplexes', '--removecomplexes', default="",
                            help='If removecomplexes="yes", indistinguishable groups will be removed before the prediction.')
        self.dnabarcoder_parser.add_argument('-redo', '--redo', default="", help='Recompute F-measure for the current parameters.')
        self.dnabarcoder_parser.add_argument('-idcolumnname', '--idcolumnname', default="ID",
                            help='the column name of sequence id in the classification file.')
        self.dnabarcoder_parser.add_argument('-display', '--display', default="",
                            help='If display=="yes" then the plot figure is displayed.')
        self.dnabarcoder_parser.add_argument('-best', action='store_true', help='Compute best similarity cut-offs for the sequences')


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

    def dnabarcoder(self, args):
        arguments = sys.argv[2:]
        if "-best" in arguments:
            arguments.remove("-best")
        #arguments = args.__dict__.items()
        #arguments = [str(arg) + " " + str(value) for arg, value in arguments if value is not None and value != ""]
        print(arguments)
        prediction_script = os.path.join(project_directory, "dnabarcoder", "prediction", "predict.py")
        arguments.insert(0, prediction_script)
        exe = sys.executable
        arguments.insert(0, exe)
        result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        print(result.stdout)
        if "-best" in sys.argv:
            arguments.remove("-best")
            arguments.append("-best")
            result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            print(result.stdout)



