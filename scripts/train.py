import ast
import os
import subprocess
import sys
from pathlib import Path
import random


script_directory = Path(__file__).parent.absolute()
project_directory = script_directory.parent.absolute()

sys.path.append(str(project_directory))

from mycoai.deep.train import DeepITSTrainer

import torch
from  mycoai.data import Data
from mycoai import utils
from mycoai.deep.models import BERT, DeepITSClassifier
from mycoai.deep.models.architectures import ResNet, CNN
from mycoai.deep.train.weight_schedules import Constant

from loggingwrapper import LoggingWrapper


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
        self.dnabarcoder_parser.add_argument('-best', '--best', action='store_true', help='Compute best similarity cut-offs for the sequences', default=False)
        self.dnabarcoder_parser.add_argument('-unique_rank', '--unique_rank', default=None, type=str, help='Select only unique sequences. If a value is also passed, unique sequences at that rank will be selected.', choices=['', 'phylum', 'class', 'order', 'family', 'genus', 'species'])


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
        '''
         This function is used to predict the similarity cut-off for sequence identification based on taxonomic classification.
         The prediction is implemented as a pipeline of three steps: (1) Select unique sqeuences at the given rank, (2) Predict the similarity cut-off for the selected sequences, and (3) Compute the best similarity cut-off for the whole dataset.
         Step 1 and 3 are optional and are only executed if -unique_rank and -best are set, respectively.
        Parameters
        ----------
        args

        Returns
        -------

        '''

        LoggingWrapper.info("Starting dnabarcoder prediction...", color="green",bold=True)
        arguments = sys.argv[2:]
        unique_fastafile_name = None
        classification_file_name = args.classification
        predict_file_extension = ".cutoffs.json"

        # -----------------select sequences if unique_rank option is set-------------------------
        if args.unique_rank is not None:
            LoggingWrapper.info("Selecting unique sequences...", color="green")
            nameExt = str(random.randint(0, 100000))
            unique_fastafile_name = args.out + "/" + Path(args.input).stem + "." + nameExt + ".fasta"
            classification_file_name = args.out + "/" + Path(args.input).stem + "." + nameExt + ".classification"
            predict_file_extension = "." + nameExt + predict_file_extension
            if args.unique_rank == "":
                unique_command_args = ["-i", args.input, "-o", unique_fastafile_name, "-c", args.classification, "-rank", args.unique_rank, "-unique", "yes"]
            else:
                unique_command_args = ["-i", args.input, "-o", unique_fastafile_name, "-c", args.classification, "-unique", "yes"]
            unique_script = os.path.join(project_directory, "dnabarcoder", "aidscripts", "selectsequences.py")
            unique_command_args.insert(0, unique_script)
            exe = sys.executable
            unique_command_args.insert(0, exe)
            unique_result = subprocess.run(unique_command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', check=True)
            if unique_result.returncode != 0:
                LoggingWrapper.error("Error while selecting unique sequences.", color="red", bold=True)
                for line in unique_result.stderr.splitlines():
                    LoggingWrapper.error(line)
                sys.exit(unique_result.returncode)
            LoggingWrapper.info("Unique sequences selected.", color="green")
            for line in unique_result.stdout.splitlines():
                LoggingWrapper.info(line)
            if (args.unique_rank != ""):
                try:
                    unique_rank_index = arguments.index("-unique_rank")
                except ValueError:
                    unique_rank_index = arguments.index("--unique_rank")
                del arguments[unique_rank_index + 1]
            try:
                arguments.remove("-unique_rank")
            except ValueError:
                arguments.remove("--unique_rank")
            try:
                input_fastfile_index = arguments.index("-i")
            except ValueError:
                input_fastfile_index = arguments.index("--input")
            arguments[input_fastfile_index + 1] = unique_fastafile_name
            try:
                input_classification_index = arguments.index("-c")
            except ValueError:
                input_classification_index = arguments.index("--classification")
            arguments[input_classification_index + 1] = classification_file_name

        # -----------------predict similarity cut-offs-------------------------
        if args.best:
            try:
                arguments.remove("-best")
            except ValueError:
                arguments.remove("--best")
        prediction_script = os.path.join(project_directory, "dnabarcoder", "prediction", "predict.py")
        arguments.insert(0, prediction_script)
        exe = sys.executable
        arguments.insert(0, exe)
        LoggingWrapper.info("Predicting similarity cut-offs...", color="green")
        predict_result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', check=True)
        if predict_result.returncode != 0:
            LoggingWrapper.error("Error while predicting similarity cut-offs.", color="red", bold=True)
            for line in predict_result.stderr.splitlines():
                LoggingWrapper.error(line)
            sys.exit(predict_result.returncode)
        LoggingWrapper.info("Similarity cut-offs predicted.", color="green")
        for line in predict_result.stdout.splitlines():
            LoggingWrapper.info(line)
        if unique_fastafile_name is not None:
            Path(unique_fastafile_name).unlink()

        # -----------------compute best similarity cut-offs if best option is set-------------------------
        if args.best:
            LoggingWrapper.info("Computing best similarity cut-offs...", color="green")
            if args.prefix == "":
                predict_file = args.out + "/" + Path(args.input).stem + predict_file_extension
            else:
                predict_file = args.out + "/" + args.prefix + predict_file_extension

            best_command_args =  ["-i", predict_file, "-o", args.out, "-prefix", args.prefix, "-c",  classification_file_name]

            best_cuttoff_script = os.path.join(project_directory, "dnabarcoder", 'prediction', 'computeBestCutoffs.py')
            best_command_args.insert(0, best_cuttoff_script)
            exe = sys.executable
            best_command_args.insert(0, exe)
            best_cuttoffs_result = subprocess.run(best_command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', check=True)
            if best_cuttoffs_result.returncode != 0:
                LoggingWrapper.error("Error while computing best similarity cut-offs.", color="red", bold=True)
                for line in best_cuttoffs_result.stderr.splitlines():
                    LoggingWrapper.error(line)
                sys.exit(best_cuttoffs_result.returncode)
            LoggingWrapper.info("Best similarity cut-offs computed.", color="green")
            for line in best_cuttoffs_result.stdout.splitlines():
                LoggingWrapper.info(line)
            if (classification_file_name != args.classification):
                Path(classification_file_name).unlink()
            LoggingWrapper.info("Prediction finished.", color="green", bold=True)



