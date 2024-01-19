# Description: This script is used to classify ITS sequences using BLAST and DeepITS.
import os
import subprocess
import sys
from pathlib import Path
from Bio import SeqIO

script_directory = Path(__file__).parent.absolute()
project_directory = script_directory.parent.absolute()

sys.path.append(str(project_directory))

from mycoai.trad import BLASTClassifier
import torch
from mycoai import utils
from loggingwrapper import LoggingWrapper


class Classify:

    def __init__(self, dnabarcoder_parser, deep_parser):
        self.dnabarcoder_parser = dnabarcoder_parser
        self.deep_parser = deep_parser

    def add_dnabarcoder_args(self):
        self.dnabarcoder_parser.add_argument('-i', '--input', required=True, help='the classified file.')
        self.dnabarcoder_parser.add_argument('-f', '--fasta', default="",
                            help='The fasta file of the sequences for saving unidentified sequences. Optional.')
        self.dnabarcoder_parser.add_argument('-c', '--classification', default="", help='the classification file in tab. format.')
        self.dnabarcoder_parser.add_argument('-r', '--reference', default="",
                            help='the reference fasta file, in case the classification of the sequences is given in the sequence headers.')
        self.dnabarcoder_parser.add_argument('-o', '--out', default="dnabarcoder", help='The output folder.')
        self.dnabarcoder_parser.add_argument('-fmt', '--inputformat', default="tab delimited",
                            help='the format of the classified file. The inputfmt can have two values "tab delimited" and "blast". The value "tab delimited" is given as default, and the "blast" fmt is the format of the BLAST output with outfmt=6.')
        self.dnabarcoder_parser.add_argument('-cutoff', '--globalcutoff', type=float, default=-1,
                            help='The global cutoff to assign the sequences to predicted taxa. If the cutoffs file is not given, this value will be taken for sequence assignment.')
        self.dnabarcoder_parser.add_argument('-confidence', '--globalconfidence', type=float, default=-1,
                            help='The global confidence to assign the sequences to predicted taxa')
        self.dnabarcoder_parser.add_argument('-rank', '--classificationrank', default="", help='the classification rank')
        self.dnabarcoder_parser.add_argument('-prefix', '--prefix', help='the prefix of output filenames')
        self.dnabarcoder_parser.add_argument('-cutoffs', '--cutoffs',
                            help='The json file containing the local cutoffs to assign the sequences to the predicted taxa.')
        self.dnabarcoder_parser.add_argument('-minseqno', '--minseqno', type=int, default=0,
                            help='the minimum number of sequences for using the predicted cut-offs to assign sequences. Only needed when the cutoffs file is given.')
        self.dnabarcoder_parser.add_argument('-mingroupno', '--mingroupno', type=int, default=0,
                            help='the minimum number of groups for using the predicted cut-offs to assign sequences. Only needed when the cutoffs file is given.')
        self.dnabarcoder_parser.add_argument('-ml', '--minalignmentlength', type=int, default=400,
                            help='Minimum sequence alignment length required for BLAST. For short barcode sequences like ITS2 (ITS1) sequences, minalignmentlength should probably be set to smaller, 50 for instance.')
        self.dnabarcoder_parser.add_argument('-saveclassifiedonly', '--saveclassifiedonly', default=False,
                            help='The option to save all (False) or only classified sequences (True) in the classification output.')
        self.dnabarcoder_parser.add_argument('-idcolumnname', '--idcolumnname', default="ID",
                            help='the column name of sequence id in the classification file.')
        self.dnabarcoder_parser.add_argument('-display', '--display', default="",
                            help='If display=="yes" then the krona html is displayed.')
        self.dnabarcoder_parser.add_argument('-search_refernce', '--search_refernce', default=None, help='The reference fasta file used in the BLAST search.')


    def add_deep_args(self):
        self.deep_parser.add_argument('--load_model', type=str, help='Path to model to load', required=True)

        self.deep_parser.add_argument('fasta_filepath',
                           help='Path to the FASTA file containing ITS sequences to classify')

        self.deep_parser.add_argument('--out',
                           default='prediction.csv',
                           type=str,
                           nargs=1,
                           help='Path to the output CSV file to save the classification results.')
        self.deep_parser.add_argument('--gpu', type=int, const=0, nargs='?',
                                      help='Use CUDA enabled GPU if available. The number indicates the GPU to use',
                                      default=None)

    def deep(self, args):
        deep_its_model = torch.load(args.load_model)
        if args.gpu is not None:
            utils.set_device('cuda:' + str(args.cuda))  # To specify GPU use
        prediction = deep_its_model.classify(args.fasta_filepath)
        prediction.to_csv(args.out)

    def is_fasta_file(self, file_path, num_records=1):
        try:
            with open(file_path, 'r') as file:
                # Use SeqIO to parse the specified number of records
                records = list(SeqIO.parse(file, 'fasta', num_records=num_records))

                # Check if at least one record was successfully parsed
                return bool(records)

        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return False
    def dnabarcoder(self, args):
        LoggingWrapper.info("Classifying sequences using dnabarcoder ...")
        arguments = sys.argv[2:]
        if (self.is_fasta_file(args.input, 1)):
            if (args.search_refernce is None):
                LoggingWrapper.error("Input is a FASTA file, but no reference file is given for BLAST search.", color="red", bold=True)
                sys.exit(1)

            search_script = os.path.join(project_directory, "dnabarcoder", "classification", "search.py")
            search_command_args = ["-i", args.input, "-r", args.search_reference, "-ml", str(args.minalignmentlength), "-o", args.out]
            search_command_args.insert(0, search_script)
            exe = sys.executable
            search_command_args.insert(0, exe)
            search_result = subprocess.run(search_command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', check=True)
            if search_result.returncode != 0:
                LoggingWrapper.error("Error while searching sequences.", color="red", bold=True)
                for line in search_result.stderr.splitlines():
                    LoggingWrapper.error(line)
                sys.exit(search_result.returncode)
            LoggingWrapper.info("Sequences search complete.", color="green")
            for line in search_result.stdout.splitlines():
                LoggingWrapper.info(line)

            classify_input = args.out + "/" + Path(args.search_reference).stem + "." + Path(args.input).stem + "_BLAST.bestmatch"
            try:
                classify_input_index = arguments.index("-i")
            except ValueError:
                classify_input_index = arguments.index("--input")
            arguments[classify_input_index + 1] = classify_input
            try:
                search_ref_index = arguments.index("-search_reference")
            except ValueError:
                search_ref_index = arguments.index("--search_reference")
            del arguments[search_ref_index + 1]
            try:
                arguments.remove("-search_reference")
            except ValueError:
                arguments.remove("--search_reference")
        LoggingWrapper.info("Classifying sequences using dnabarcoder ...", color="green")
        classify_script = os.path.join(project_directory, "dnabarcoder", "classification", "classify.py")
        classify_command_args = arguments
        classify_command_args.insert(0, classify_script)
        exe = sys.executable
        classify_command_args.insert(0, exe)
        classify_result = subprocess.run(classify_command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', check=True)
        if classify_result.returncode != 0:
            LoggingWrapper.error("Error while classifying sequences.", color="red", bold=True)
            for line in classify_result.stderr.splitlines():
                LoggingWrapper.error(line)
            sys.exit(classify_result.returncode)
        LoggingWrapper.info("Sequences classification complete.", color="green")
        for line in classify_result.stdout.splitlines():
            LoggingWrapper.info(line)
        LoggingWrapper.info("Classifying sequences using dnabarcoder complete.", color="green", bold=True)