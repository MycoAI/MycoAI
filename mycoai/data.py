'''Contains mycoai Dataset class and data preprocessing.'''

import torch
import random
import pandas as pd
import numpy as np
from . import utils, plotter, encoders
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    '''Holds ITS data in sequence & taxonomy tensors.
  
    Attributes
    ----------
    sequences: torch.Tensor
        [n,4,max_length]-dimensional Tensor containing encoded sequence data
    taxonomies: torch.Tensor
        [n,6]-dimensional Tensor with encoded taxonomic labels (on 6 levels)
    dna_encoder: DNAEncoder
        Class that was used to generate the sequences tensor
    tax_encoder: 
        Class that was used to generate the taxonomies tensor'''

    def __init__(self, sequences=None, taxonomies=None, dna_encoder=None, 
                 tax_encoder=None, filepath=None):
        '''Initializes Dataset object from tensors or imports from filepath'''

        if filepath is not None:
            self.import_data(filepath)
        else:
            self.sequences = sequences
            self.taxonomies = taxonomies
            self.dna_encoder = dna_encoder
            self.tax_encoder = tax_encoder

    def __getitem__(self, idx):
        return self.sequences[idx], self.taxonomies[idx]

    def __len__(self):
        return self.sequences.shape[0]

    # NOTE This should be depracated (or changed) once data sampler is in use
    def weighted_loss(self, loss_function):
        '''Returns list of weighted loss'''
        filtered = ([self.taxonomies[:,i]
                    [self.taxonomies[:,i] != utils.UNKNOWN_INT] 
                    for i in range(len(utils.LEVELS))])
        sizes = [torch.bincount(taxonomies).to(utils.DEVICE) 
                 for taxonomies in filtered]
        losses = ([loss_function(weight=1/sizes[l], 
                                 ignore_index=utils.UNKNOWN_INT) 
                   for l in range(len(utils.LEVELS))])
        return losses
    
    def export_data(self, export_path):
        '''Saves sequences, taxonomies, and encoders to file.'''
        content = {'sequences':    self.sequences,
                   'taxonomies':   self.taxonomies,
                   'dna_encoder':  self.dna_encoder,
                   'tax_encoder':  self.tax_encoder}
        torch.save(content, export_path)

    def import_data(self, import_path):
        '''Imports encoded sequences and taxonomies'''
        content = torch.load(import_path)
        self.sequences = content['sequences']
        self.taxonomies = content['taxonomies']
        self.dna_encoder = content['dna_encoder']
        self.tax_encoder = content['tax_encoder']

    def labels_report(self):
        '''Prints the number of classes to predict on each level.'''
        print("No. of to-be-predicted classes:")
        num_classes = [len(self.tax_encoder.labels[i]) for i in range(6)]
        table = pd.DataFrame([['Classes (#)'] + num_classes], 
                             columns=[''] + utils.LEVELS)
        table = table.set_index([''])
        print(table)

    def unknown_labels_report(self):
        '''Prints the number of unrecognized target labels on each level.'''
        print("No. unrecognized target labels:")
        exact = torch.count_nonzero(torch.eq(self.taxonomies, 
                                             utils.UNKNOWN_INT), dim=0)
        perc = 100 * exact / len(self.taxonomies)
        table = pd.DataFrame([['Exact (#)'] + list(exact.numpy()), 
                              ['Perc. (%)'] + list(perc.numpy())], 
                             columns=[''] + utils.LEVELS)
        table = table.set_index([''])
        table = table.round(1)
        print(table)


class DataPrep: 
    '''ITS data preprocessing and preparing for use by mycoai
    
    Attributes
    ----------
    data: pd.DataFrame
        Holds dataframe containing taxonomies and sequences, with columns
        'phylum', 'class', 'order', 'genus', 'family', 'species', and 'sequence'
    '''

    def __init__(self, filename, parser='unite', allow_duplicates=False):
        '''Loads data into DataPrep object, using parser adjusted to data format
        
        Parameters
        ----------
        filename: str
            Path of to-be-loaded file
        parser: function | 'unite'
            Function that parses input and outputs a dataframe (default is 
            'unite') at least consisting of columns 'phylum', 'class', 'order', 
            'genus', 'family', 'species', and 'sequence'. Rows of this dataframe 
            should be of type str, uncertain labels should be given the value of 
            `utils.UNKNOWN_STR`. 
        allow_duplicates: bool
            Drops duplicate entries if False (default is False)'''
        
        parsers = {'unite':self.unite_parser}
        parser = parser if type(parser) != str else parsers[parser]
        data = parser(filename)
        if not allow_duplicates:
            data.drop_duplicates(subset=['sequence', 'species'], inplace=True)
        self.data = data
        if utils.VERBOSE > 0:
            print(len(self.data), "samples loaded into dataframe.")
            plotter.counts_barchart(self.data)
            plotter.counts_boxplot(self.data)
            if utils.VERBOSE == 2:
                self.length_report()
                plotter.counts_sunburstplot(self.data)

    def encode_dataset(self, dna_encoder=encoders.FourDimDNA(),tax_encoder=None, 
                    export_path=None):
        '''Converts data into a mycoai Dataset object
        
        Parameters
        ----------
        dna_encoder: DNAEncoder
            Class used to generate the sequences tensor (default is FourDimDNA)
        tax_encoder: 
            Class used to generate the taxonomies tensor, can use existing one-hot
            encoding scheme, otherwise creates a new TaxonEncoder class 
        export_path:
            Path to save encodings to (default is None)'''

        if utils.VERBOSE > 0:
            print("Encoding the data into network-readable format...")

        # Initialize encoding methods
        tax_encoder = (encoders.TaxonEncoder(self.data) if tax_encoder == None 
                    else tax_encoder)
        # Sorting is a prerequisite for our taxonomic encoding method
        data = self.data.sort_values(by=utils.LEVELS).reset_index(drop=True)

        # Loop through the data, encode sequences and labels 
        sequences, taxonomies = [], []
        for index, row in tqdm(data.iterrows()):
            sequences.append(dna_encoder.encode(row))
            taxonomies.append(tax_encoder.encode(row))

        tax_encoder.train = False # To apply taxon encoder later

        # Convert to tensors and store
        sequences = torch.tensor(sequences, dtype=torch.float32)
        sequences = torch.transpose(sequences, 1, 2)
        taxonomies = torch.tensor(taxonomies,dtype=torch.int64)
        data = Dataset(sequences, taxonomies, dna_encoder, tax_encoder)
        if utils.VERBOSE > 0:
            data.labels_report() 
            data.unknown_labels_report() if tax_encoder is not None else 0
        data.export_data(export_path) if export_path is not None else 0
        
        return data 

    def unite_parser(self, filename):
        '''Reads a Unite FASTA file into a Pandas dataframe'''

        # NOTE using .readlines(x) limits the data to x bytes, use for debugging
        unite_file = open(filename).readlines()

        data = []
        if utils.VERBOSE > 0:
            print("Parsing UNITE data...")
        for i in tqdm(range(0,len(unite_file),2)):
            header = unite_file[i][:-1].split('|') # Remove \n and split columns
            seq = unite_file[i+1][:-1] # Sequence on the next line, remove \n
            taxonomy = header[1] # Retrieve taxonomy data      
            taxonomy, species = taxonomy.split(';s__')
            taxonomy, genus = taxonomy.split(';g__')
            taxonomy, family = taxonomy.split(';f__')
            taxonomy, order = taxonomy.split(';o__')
            taxonomy, classs = taxonomy.split(';c__')
            taxonomy, phylum = taxonomy.split(';p__')
            sh = header[2] # Get SH 
            data_row = [sh, phylum, classs, order, family, genus, species, seq]
            data.append(data_row)

        data = pd.DataFrame(data, columns=['sh'] + utils.LEVELS + ['sequence'])
        data = self.unite_label_uncertain(data)

        return data

    def unite_label_uncertain(self, data):
        '''Replaces labels with 'Incertae_sedis' and 'GS' taxons.'''
        
        levels = utils.LEVELS
        for i in range(6):
            # Replace all deeper levels if we find an uncertain prediction 
            # (this might throw away some data but is to preserve consistency)
            repl = [utils.UNKNOWN_STR for i in range(len(levels[i:]))]
            data.loc[data[levels[i]].str[-14:]=='Incertae_sedis', 
                     levels[i:]] = repl
            data.loc[data[levels[i]].str[:2]=='GS', levels[i:]] = repl

        # Simply remove the data row if all taxon levels are unknown
        return data[data['phylum'] != utils.UNKNOWN_STR]
    
    def length_report(self):
        '''Prints min, max, median, and std of sequences in dataframe.'''

        minn = self.data['sequence'].str.len().min()
        maxx = self.data['sequence'].str.len().max()
        median = self.data['sequence'].str.len().median()
        std = self.data['sequence'].str.len().std()
        print("Sequence lengths (min, max, median, std):", 
              minn, maxx, median, std)

    def class_filter(self, level, min_samples=0, max_samples=np.inf, 
                     max_classes=np.inf):
        '''Retains at most max_samples sequences at specified taxon level
        for which at least min_samples are available in that class.
        Ensures perfect class balance when min_samples == max_samples.
        Randomly selects a max_classes number of classes.'''

        data = self.data.groupby(level).filter(lambda x: len(x) > min_samples)
        groups = [group for _, group in data.groupby(level)]
        data = pd.concat(random.sample(groups, min(max_classes, len(groups))))
        # data = data[data[level] != unidentified_label] # NOTE optional
        # Randomly select out of the max_samples
        data = data.sample(frac=1) 
        data = data.groupby(level).head(max_samples).reset_index(drop=True)

        if utils.VERBOSE > 0:
            print(len(data), "sequences retained after class count filter")
            plotter.counts_boxplot(data, id='class_filtered')
            if utils.VERBOSE == 2:
                plotter.counts_sunburstplot(data, id='class_filtered')

        self.data = data
        return self