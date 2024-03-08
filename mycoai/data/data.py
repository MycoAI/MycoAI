'''Sequence data reading, preprocessing, encoding, and writing'''

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
from Bio import SeqIO


class Data: 
    '''Data container that parses and filters sequence data and encodes it into 
    the network-readable TensorData format.
    
    Attributes
    ----------
    data: pd.DataFrame
        Holds dataframe containing taxonomies and sequences, with columns 'id',
        'phylum', 'class', 'order', 'genus', 'family', 'species', and 'sequence'
    name: str
        Name of dataset.
    '''

    def __init__(self, filepath, tax_parser='unite', allow_duplicates=False,
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
        
        # Small hack to allow creating a Data object from a pd.DataFrame
        if type(filepath) == pd.DataFrame:
            self.data = filepath
            self.name = name
            return

        tax_parsers = {'unite':  self.unite_parser}
        if type(tax_parser) == str:
            tax_parser = tax_parsers[tax_parser]
        data = self.read_fasta(filepath, tax_parser=tax_parser)

        if not allow_duplicates:
            if tax_parser is not None:
                data.drop_duplicates(subset=['sequence','species'],inplace=True)
            else:
                data.drop_duplicates(subset=['sequence'], inplace=True)

        self.data = data
        self.name = utils.filename_from_path(filepath) if name is None else name

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
                       export_path=None):
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
        export_path: list | str
            Path to save encodings to (default is None)'''

        if utils.VERBOSE > 0:
            print("Encoding the data into network-readable format...")

        # INITIALIZING
        dna_encs = {'4d':            encoders.FourDimDNA,
                    'kmer-tokens':   encoders.KmerTokenizer,
                    'kmer-onehot':   encoders.KmerOneHot,
                    'kmer-spectral': encoders.KmerSpectrum,
                    'bpe':           encoders.BytePairEncoder}
        tax_encs = {'categorical':   encoders.TaxonEncoder}
        if type(dna_encoder) == str:
            if dna_encoder == 'bpe':
                dna_encoder = dna_encs[dna_encoder](self)
            else:
                dna_encoder = dna_encs[dna_encoder]() 
        if not self.labelled():
            print("No data labels have been imported.") 
            print("If the data is labelled, specify tax_parser in init.")
            tax_encoder = type('placeholder', (), 
                               {'encode': lambda _, x: torch.zeros(1),
                                'encode_fast': 
                                    (lambda _, x: 
                                     torch.zeros((len(self.data), 1))),
                                'train': False})()
        elif type(tax_encoder) == str:
            tax_encoder = tax_encs[tax_encoder](self)

        # ENCODING (of sequences and their labels)
        if (type(dna_encoder) == encoders.BytePairEncoder and 
            not tax_encoder.train): # Check if fast encoding is available
            taxonomies = tax_encoder.encode_fast(self)
            sequences = dna_encoder.encode_fast(self)
        else: # If not, iterrate over rows and encode row-by-row
            sequences, taxonomies = [], []
            for index, row in tqdm(self.data.iterrows()):
                sequences.append(dna_encoder.encode(row['sequence']))
                taxonomies.append(tax_encoder.encode(row))
            if type(tax_encoder) == encoders.TaxonEncoder and tax_encoder.train:
                tax_encoder.finish_training()
            sequences = torch.stack(sequences)
            taxonomies = torch.stack(taxonomies)
        
        # Create TensorData object
        data = TensorData(sequences, taxonomies, dna_encoder, tax_encoder, 
                          self.name)
        
        if utils.VERBOSE > 0 and self.labelled():
            data.labels_report() 
            data.unknown_labels_report() if tax_encoder is not None else 0

        data.export_data(export_path) if export_path is not None else 0
        return data
    
    def train_valid_split(self, valid_split, export_fasta=False):
        '''Splits up the mycoai.Data object into a train and validation split.
        Writes fasta file if export_fasta is True. If type(export_fasta)==list,
        will write train and valid data to specified filepaths.'''

        train, valid = train_test_split(self.data, test_size=valid_split)
        train = Data(train, name=self.name + f' ({1-valid_split})')
        valid = Data(valid, name=self.name + f' ({valid_split})')

        if export_fasta:
            if type(export_fasta) == list:
                train.export_fasta(export_fasta[0])
                valid.export_fasta(export_fasta[1])
            else:
                train.export_fasta(train.name + '.fasta')
                valid.export_fasta(valid.name + '.fasta')

        return train, valid
    
    def read_fasta(self, filename, tax_parser=None):
        '''Reads a FASTA file into a Pandas dataframe'''

        data = []
        seqrecords=list(SeqIO.parse(filename, "fasta"))
        for seqrecord in tqdm(seqrecords):
            header=seqrecord.description
            seq=str(seqrecord.seq).upper()
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

    def unite_parser(self, fasta_header):
        '''Parses FASTA headers using the UNITE format to extract taxonomies'''

        # Retrieving taxonomies
        fasta_header = fasta_header.split('|')
        id = fasta_header[0]
        taxonomy = fasta_header[1] # Retrieve taxonomy data      
        taxonomy, species = taxonomy.split(';s__')
        taxonomy, genus = taxonomy.split(';g__')
        taxonomy, family = taxonomy.split(';f__')
        taxonomy, order = taxonomy.split(';o__')
        taxonomy, classs = taxonomy.split(';c__')
        taxonomy, phylum = taxonomy.split(';p__')
        data_row = [id, phylum, classs, order, family, genus, species]

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
    
    def label_cascade(self):
        '''Creates new child labels for samples with parent labels that only 
        have unidentified children in the dataset. E.g. when a certain genus 
        never has identified species, create a new species for this genus such 
        that it won't be missed when training only on species-level.'''

        print(self.num_classes_per_level())
        self.unknown_labels_report()
        print('start')
        for i, lvl in enumerate(utils.LEVELS[:5]): # From phylum to genus
            # Calculate number of unique subclasses
            next_lvl = utils.LEVELS[i+1] 
            count = self.data[[lvl, next_lvl]].groupby(lvl, 
                                                       as_index=False).nunique()
            # Find parents with 1 child on next lvl
            one_child = count[(count[next_lvl] == 1) & 
                              (count[lvl] != utils.UNKNOWN_STR)][lvl] 
            # Find rows for which the only child is the unidentified class
            only_unk = self.data[(self.data[lvl].isin(one_child)) &
                                 (self.data[next_lvl] == utils.UNKNOWN_STR)]
            # Create new labels (parent + class indicator) for these rows
            new_labels = only_unk[lvl] + '_' + next_lvl[0] 
            self.data.loc[only_unk.index, next_lvl] = new_labels
        print('finish')
        print(self.num_classes_per_level())
        self.unknown_labels_report()

        return self.data
    
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

        if not self.labelled():
            raise AttributeError('No data labels have been imported.')

        print("Kullback-leibler divergence from ideal distribution:")
        klds = []
        for lvl in utils.LEVELS:
            data = self.data[(self.data[lvl] != utils.UNKNOWN_STR)]
            actual = data.groupby([lvl])['id'].count().values
            ideal = [1/data[lvl].nunique()]*data[lvl].nunique()
            kld = scipy.stats.entropy(actual, ideal)
            klds.append(kld)
            print(lvl + ":", np.round(kld,2))
        print('-------------')
        print('average:', np.round(np.average(klds),2))

    def class_filter(self, level, min_samples=0, max_samples=np.inf, 
                     max_classes=np.inf, remove_unidentified=False):
        '''Retains at most max_samples sequences at specified taxon level
        for which at least min_samples are available in that class.
        Ensures perfect class balance when min_samples == max_samples.
        Randomly selects a max_classes number of classes.'''

        if not self.labelled():
            raise AttributeError('No data labels have been imported.')

        if remove_unidentified:
            data = self.data[self.data[level] != utils.UNKNOWN_STR]
        else:
            data = self.data
        data = data.groupby(level).filter(lambda x: len(x) > min_samples)
        groups = [group for _, group in data.groupby(level)]
        data = pd.concat(random.sample(groups, min(max_classes, len(groups))))
        # Randomly select out of the max_samples
        data = data.sample(frac=1) 
        data = data.groupby(level).head(max_samples).reset_index(drop=True)

        if utils.VERBOSE > 0:
            print(len(data), "sequences retained after class count filter")

        self.data = data
        return self

    def remove_level_filter(self, level, mask='Dummy'):
        '''Removes all labels (set to unidentified) at specified level(s)'''
        if type(level) == str:
            level = [level]
        for lvl in level:
            self.data[lvl] = mask
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

    def labels_report(self):
        '''Prints the number of classes to predict on each level.'''
        print("No. of to-be-predicted classes:")
        num_classes = self.num_classes_per_level()
        table = pd.DataFrame([['Classes (#)'] + num_classes], 
                             columns=[''] + utils.LEVELS)
        table = table.set_index([''])
        print(table)

    def unknown_labels_report(self):
        '''Prints the number of unrecognized target labels on each level.'''
        print("No. unrecognized target labels:")
        exact = (self.data[utils.LEVELS] == utils.UNKNOWN_STR).sum()
        perc = 100 * exact / len(self.data)
        table = pd.DataFrame([['Exact (#)'] + list(exact.values), 
                              ['Perc. (%)'] + list(perc.values)], 
                             columns=[''] + utils.LEVELS)
        table = table.set_index([''])
        table = table.round(1)
        print(table)

    def num_classes_per_level(self):
        '''Number of classes per taxonomic level'''

        if not self.labelled():
            raise AttributeError('No data labels have been imported.')

        output = []
        for lvl in range(6):
            known_data = self.data[
                self.data[utils.LEVELS[lvl]] != utils.UNKNOWN_STR
            ][utils.LEVELS[lvl]]
            output.append(known_data.nunique(0))
        return output

    def get_class_size(self, mode):
        '''Min/max/med number of examples per class per taxonomic level'''

        if not self.labelled():
            raise AttributeError('No data labels have been imported.')
        
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

        if self.labelled():
            labels_config = {
                'classes_per_lvl':  self.num_classes_per_level(),
                'min_class_size':   self.get_class_size('min'),
                'max_class_size':   self.get_class_size('max'),
                'med_class_size':   self.get_class_size('med')
            }
        else:
            labels_config = {}

        return {
            'name':             self.name,
            'num_examples':     len(self.data),
            **labels_config
        }