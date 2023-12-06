'''ITS sequence data reading, preprocessing, encoding, and writing'''

import scipy
import torch
import random
import pandas as pd
import numpy as np
from mycoai.data import encoders
from mycoai import utils, plotter
from .tensor_data import TensorData
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Data:
    '''ITS data preprocessor and container

    Attributes
    ----------
    data: pd.DataFrame
        Holds dataframe containing taxonomies and sequences, with columns 'id',
        'phylum', 'class', 'order', 'genus', 'family', 'species', and 'sequence'
    '''

    def __init__(self, filename, tax_parser='unite', allow_duplicates=False,
                 name=None):
        '''Loads data into Data object, using parser adjusted to data format

        Parameters
        ----------
        filename: str
            Path of to-be-loaded file in FASTA format
        tax_parser: function | 'unite'
            Function that parses the FASTA headers and extracts the taxonomic
            labels on all six levels (from phylum to species). If None, no
            labels will be extracted. If 'unite', will follow the UNITE format.
            Also supports user-custom functions, as long as they return a list
            following the format: [id, phyl, clas, ord, fam, gen, spec].
        allow_duplicates: bool
            Drops duplicate entries if False (default is False)
        name: str
            Name of dataset. Will be inferred from filename if None.'''

        tax_parsers = {'unite':  self.unite_parser}
        if type(tax_parser) == str:
            tax_parser = tax_parsers[tax_parser]
        data = self.read_fasta(filename, tax_parser=tax_parser)

        if not allow_duplicates:
            if tax_parser is not None:
                data.drop_duplicates(subset=['sequence','species'],inplace=True)
            else:
                data.drop_duplicates(subset=['sequence'], inplace=True)

        self.data = data
        self.name = utils.filename_from_path(filename) if name is None else name

        if utils.VERBOSE > 0:
            print(len(self.data), "samples loaded into dataframe.")
            if tax_parser is not None:
                plotter.counts_barchart(self)
                plotter.counts_boxplot(self)
            if utils.VERBOSE == 2:
                self.length_report()
                if tax_parser is not None:
                    plotter.counts_sunburstplot(self)

    def encode_dataset(self, dna_encoder, tax_encoder='categorical',
                       valid_split=0.0, export_path=None, k=4):
        '''Converts data into a mycoai TensorData object

        Parameters
        ----------
        dna_encoder: DNAEncoder | str
            Specifies the encoder used for generating the sequence tensor.
            Can be an existing object, or one of ['4d', 'kmer-tokens',
            'kmer-onehot', 'kmer-spectral', 'bpe'], which will initialize an
            encoder of that type
        tax_encoder: TaxonEncoder | str
            Specifies the encoder used for generating the taxonomies tensor.
            Can be an existing object, or 'categorical', which will
            initialize an encoder of that type  (default is 'categorical')
        valid_split: float
            If >0, will split data in valid/train at ratio (default is 0.0)
        export_path: list | str
            Path to save encodings to. Needs to be type `list' (length 2) when
            valid_split > 0. First element of that list should refer to the
            train export_path, second to the valid path (default is None)'''

        if utils.VERBOSE > 0:
            print("Encoding the data into network-readable format...")

        # Initialize encoding methods
        dna_encs = {'4d':            encoders.FourDimDNA,
                    'kmer-tokens':   encoders.KmerTokenizer,
                    'kmer-onehot':   encoders.KmerOneHot,
                    'kmer-spectral': encoders.KmerSpectrum,
                    'bpe':           encoders.BytePairEncoder}
        tax_encs = {'categorical':   encoders.TaxonEncoder}
        if type(dna_encoder) == str:
            if dna_encoder == 'bpe':
                dna_encoder = dna_encs[dna_encoder](self)
            elif dna_encoder == '4d':
                dna_encoder = dna_encs[dna_encoder]()
            else:
                dna_encoder = dna_encs[dna_encoder](k)
        if not self.labelled():
            print("No data labels have been imported.")
            print("If the data is labelled, specify tax_parser in init.")
            tax_encoder = type('placeholder', (),
                               {'encode': lambda self, x: torch.zeros(1)})()
        elif type(tax_encoder) == str:
            tax_encoder = tax_encs[tax_encoder](self.data)

        if valid_split > 0:
            train_df, valid_df = train_test_split(self.data,
                                                  test_size=valid_split)
            data = self._encode_subset(train_df, dna_encoder, tax_encoder,
                                       self.name + f' ({1-valid_split})')
            valid_data = self._encode_subset(valid_df, dna_encoder, tax_encoder,
                                             self.name + f' ({valid_split})')
        else:
            data = self._encode_subset(self.data, dna_encoder, tax_encoder,
                                                                      self.name)

        if utils.VERBOSE > 0 and self.labelled():
            data.labels_report()
            data.unknown_labels_report() if tax_encoder is not None else 0

        if valid_split > 0:
            if export_path is not None:
                data.export_data(export_path[0])
                valid_data.export_data(export_path[1])
            return data, valid_data
        else:
            data.export_data(export_path) if export_path is not None else 0
            return data

    @staticmethod
    def _encode_subset(dataframe, dna_encoder, tax_encoder, name):
        '''Encodes dataframe given the specified encoders'''

        # Loop through the data, encode sequences and labels
        sequences, taxonomies = [], []
        for index, row in tqdm(dataframe.iterrows()):
            sequences.append(dna_encoder.encode(row['sequence']))
            taxonomies.append(tax_encoder.encode(row))

        if type(tax_encoder) == encoders.TaxonEncoder and tax_encoder.train:
            tax_encoder.finish_training()

        # Convert to tensors and store
        sequences = torch.stack(sequences)
        taxonomies = torch.stack(taxonomies)
        data = TensorData(sequences, taxonomies, dna_encoder, tax_encoder, name)

        return data

    def read_fasta(self, filename, tax_parser=None):
        '''Reads a FASTA file into a Pandas dataframe'''

        # NOTE using .readlines(x) limits the data to x bytes, use for debugging
        unite_file = open(filename).readlines()

        data = []
        for i in tqdm(range(0,len(unite_file),2)):
            header = unite_file[i][:-1] # Remove \n
            seq = unite_file[i+1][:-1] # Sequence on the next line, remove \n
            if tax_parser is not None:
                data_row = tax_parser(header) + [seq]
            else:
                data_row = [header, seq]
            data.append(data_row)

        columns = ['id', 'sequence']
        if tax_parser is not None:
            columns = ['id'] + utils.LEVELS + ['sequence']
        data = pd.DataFrame(data, columns=columns)

        return data

    def export_fasta(self, filepath):
        '''Exports this Data object to a FASTA file'''

        file = open(filepath, 'w')
        for i, row in self.data.iterrows():
            header = '>|k__Fungi;p__' + row['phylum']
            header += ';c__' + row['class']
            header += ';o__' + row['order']
            header += ';f__' + row['family']
            header += ';g__' + row['genus']
            header += ';s__' + row['species']
            header += '|' + row['id']
            file.write(header + '\n')
            file.write(row['sequence'] + '\n')

        file.close()

    def mycoai_parser(self, fasta_header):
        '''Parses FASTA headers using the MycoAI format to extract taxonomies'''
        return fasta_header[1:-1].split(";")

    def unite_parser(self, fasta_header):
        '''Parses FASTA headers using the UNITE format to extract taxonomies'''

        # Retrieving taxonomies
        fasta_header = fasta_header.split('|')
        sh = fasta_header[2]
        taxonomy = fasta_header[1] # Retrieve taxonomy data
        taxonomy, species = taxonomy.split(';s__')
        taxonomy, genus = taxonomy.split(';g__')
        taxonomy, family = taxonomy.split(';f__')
        taxonomy, order = taxonomy.split(';o__')
        taxonomy, classs = taxonomy.split(';c__')
        taxonomy, phylum = taxonomy.split(';p__')
        data_row = [sh, phylum, classs, order, family, genus, species]

        return self.unite_label_preprocessing(data_row)

    def unite_label_preprocessing(self, data_row):
        '''Cleans up labels in the UNITE dataset, labels uncertain taxons'''

        # Handling uncertain taxons
        for i in range(1,7):
            if ((data_row[i][-14:].lower() == 'incertae_sedis') or
                (data_row[i][:2] == 'GS') or
                (i == 6 and data_row[i][-3:] == '_sp')):
                    data_row[i] = utils.UNKNOWN_STR
        # Removing extra info on species level
        for addition in ['_subsp\\.', '_var\\.', '_f\\.']:
            data_row[6] = data_row[6].split(addition)[0]

        return data_row

    def length_report(self):
        '''Prints min, max, median, and std of sequences in dataframe.'''

        minn = self.data['sequence'].str.len().min()
        maxx = self.data['sequence'].str.len().max()
        median = self.data['sequence'].str.len().median()
        std = self.data['sequence'].str.len().std()
        print("Sequence lengths (min, max, median, std):",
              minn, maxx, median, std)

    def class_imbalance_report(self):
        '''Reports class imbalance per taxon level by calculating the kullback-
        leibler divergence from a perfectly balanced distribution'''

        print("Kullback-leibler divergence from ideal distribution:")
        klds = []
        for lvl in utils.LEVELS:
            data = self.data[(self.data[lvl] != utils.UNKNOWN_STR)]
            actual = data.groupby([lvl])['sequence'].count().values
            ideal = [1/data[lvl].nunique()]*data[lvl].nunique()
            kld = scipy.stats.entropy(actual, ideal)
            klds.append(kld)
            print(lvl + ":", np.round(kld,2))
        print('-------------')
        print('average:', np.round(np.average(klds),2))

    def class_filter(self, level, min_samples=0, max_samples=np.inf,
                     max_classes=np.inf):
        '''Retains at most max_samples sequences at specified taxon level
        for which at least min_samples are available in that class.
        Ensures perfect class balance when min_samples == max_samples.
        Randomly selects a max_classes number of classes.'''

        data = self.data.groupby(level).filter(lambda x: len(x) > min_samples)
        groups = [group for _, group in data.groupby(level)]
        data = pd.concat(random.sample(groups, min(max_classes, len(groups))))
        # Randomly select out of the max_samples
        data = data.sample(frac=1)
        data = data.groupby(level).head(max_samples).reset_index(drop=True)

        if utils.VERBOSE > 0:
            print(len(data), "sequences retained after class count filter")

        self.data = data
        return self

    def sequence_quality_filter(self, tolerance=0.05):
        '''Removes sequences with more than tolerated no. of uncertain bases'''
        certain = self.data['sequence'].str.count("A|C|T|G")
        length = self.data['sequence'].str.len()
        ratio = (length - certain) / length
        self.data = self.data[(ratio < tolerance)]
        if utils.VERBOSE > 0:
            print(len(self.data), "sequences retained after quality filter")
        return self

    def sequence_length_filter(self, tolerance=4):
        '''Removes sequences with that fall outside of the tolerated range.
        
        Parameters
        ----------
        tolerance: int | list | range
            Tolerated range of lengths. In case of an integer, the tolerated 
            range is defined as the sequences that fall within the specified 
            number of standard deviations from the mean length.'''
        
        if type(tolerance) != int:
            self.data = self.data[
                (self.data['sequence'].str.len() >= tolerance[0]) &
                (self.data['sequence'].str.len() < tolerance[-1])]
        else:
            length = self.data['sequence'].str.len()
            std = length.std()
            avg = length.mean()
            self.data = self.data[
                (np.abs(self.data['sequence'].str.len() - avg) < tolerance*std)]
        if utils.VERBOSE > 0:
            print(len(self.data), "sequences retained after length filter")
        return self

    def labelled(self):
        '''Returns True if all 6 taxonomic labels are available'''
        for lvl in utils.LEVELS:
            if lvl not in self.data.columns:
                return False
        return True

    def num_classes_per_level(self):
        '''Number of classes per taxonomic level'''
        output = []
        for lvl in range(6):
            known_data = self.data[
                self.data[utils.LEVELS[lvl]] != utils.UNKNOWN_STR
            ][utils.LEVELS[lvl]]
            output.append(known_data.nunique(0))
        return output

    def get_class_size(self, mode):
        '''Min/max/med number of examples per class per taxonomic level'''
        output = []
        for lvl in range(6):
            known_data = self.data[
                self.data[utils.LEVELS[lvl]] != utils.UNKNOWN_STR
            ]
            bins = known_data.groupby(utils.LEVELS[lvl]).size().values
            if mode == 'min':
                output.append(bins.min())
            elif mode == 'max':
                output.append(bins.max())
            else: # mode == 'med'
                output.append(np.median(bins))
        return output

    def get_config(self):
        '''Returns configuration dictionary of this object instance.'''
        return {
            'name':             self.name,
            'num_examples':     len(self.data),
            'classes_per_lvl':  self.num_classes_per_level(),
            'min_class_size':   self.get_class_size('min'),
            'max_class_size':   self.get_class_size('max'),
            'med_class_size':   self.get_class_size('med')
        }