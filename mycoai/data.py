'''Contains mycoai Dataset class and data preprocessing.'''

import scipy
import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as tud
from . import utils, plotter, encoders
from sklearn.model_selection import train_test_split
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
                 tax_encoder=None, name=None, filepath=None):
        '''Initializes Dataset object from tensors or imports from filepath'''

        if filepath is not None:
            self.import_data(filepath)
        else:
            self.sequences = sequences
            self.taxonomies = taxonomies
            self.dna_encoder = dna_encoder
            self.tax_encoder = tax_encoder
            self.name = name

    def __getitem__(self, idx):
        return self.sequences[idx], self.taxonomies[idx]

    def __len__(self):
        return self.sequences.shape[0]
    
    def export_data(self, export_path):
        '''Saves sequences, taxonomies, and encoders to file.'''
        content = {'sequences':    self.sequences,
                   'taxonomies':   self.taxonomies,
                   'dna_encoder':  self.dna_encoder,
                   'tax_encoder':  self.tax_encoder,
                   'name':         self.name}
        torch.save(content, export_path)

    def import_data(self, import_path):
        '''Imports encoded sequences and taxonomies'''
        content = torch.load(import_path)
        self.sequences = content['sequences']
        self.taxonomies = content['taxonomies']
        self.dna_encoder = content['dna_encoder']
        self.tax_encoder = content['tax_encoder']
        self.name = content['name']

    def get_config(self):
        '''Returns configuration dictionary of this object instance.'''
        return {
            'name':             self.name,
            'num_examples':     self.taxonomies.shape[0],
            'classes_per_lvl':  self.num_classes_per_level(),
            'min_class_size':   self.get_class_size('min'),
            'max_class_size':   self.get_class_size('max'),
            'med_class_size':   self.get_class_size('med')
        }
    
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
        exact = torch.count_nonzero(torch.eq(self.taxonomies, 
                                             utils.UNKNOWN_INT), dim=0)
        perc = 100 * exact / len(self.taxonomies)
        table = pd.DataFrame([['Exact (#)'] + list(exact.numpy()), 
                              ['Perc. (%)'] + list(perc.numpy())], 
                             columns=[''] + utils.LEVELS)
        table = table.set_index([''])
        table = table.round(1)
        print(table)

    def num_classes_per_level(self):
        '''Number of classes per taxonomic level'''
        return [len(self.tax_encoder.lvl_encoders[i].classes_)for i in range(6)]

    def get_class_size(self, mode):
        '''Min/max/med number of examples per class per taxonomic level'''
        output = []
        for lvl in range(6):
            bins = torch.bincount(self.taxonomies[:,lvl]
                                  [self.taxonomies[:,lvl] != utils.UNKNOWN_INT])
            bins = bins[bins != 0]
            if mode == 'min':
                output.append(bins.min().item())
            elif mode == 'max':
                output.append(bins.max().item())
            else: # mode == 'med'
                output.append(bins.median().item())
        return output
        
    def weighted_loss(self, loss_function, sampler=None):
        '''Returns list of weighted loss (weighted by reciprocal of class size).
        If sampler is provided, will correct for the expected data distribution
        (on all taxonomic levels) given the specified sampler.'''

        dist = None
        losses = []
        for lvl in range(5,-1,-1): # Loop backwards throug levels
            num_classes = len(self.tax_encoder.lvl_encoders[lvl].classes_)
            
            # Weigh by reciprocal of class size when no sampler in effect
            if sampler is None or lvl > sampler.lvl:   
                filtered = (self.taxonomies[:,lvl] # Filter for known entries
                            [self.taxonomies[:,lvl] != utils.UNKNOWN_INT])
                sizes = torch.bincount(filtered, minlength=num_classes) # Count
                sizes = sizes.to(utils.DEVICE)
                loss = loss_function(weight=1/sizes, # Take reciprocal
                                     ignore_index=utils.UNKNOWN_INT)
            
            # No weights at level which has a weighted sampler (perfect balance)
            elif lvl == sampler.lvl:
                loss = loss_function(ignore_index=utils.UNKNOWN_INT)
                # Initialize equally distributed distribution
                dist = torch.ones(num_classes)/num_classes
            
            # Calculate effect of weighted sampler on parent levels...
            else: # ... by inferring what the parent distribution will be
                dist = dist @ self.tax_encoder.inference_matrices[lvl].to('cpu')
                # Account for some samples that were unknown at sampler level...
                unknown = (self.taxonomies[:,lvl]
                          [self.taxonomies[:,sampler.lvl]==utils.UNKNOWN_INT])
                n_rows_unknown = len(unknown) # Calculate amount
                # ... and extract what class they are on parent level
                filtered = unknown[unknown != utils.UNKNOWN_INT]
                add_random = torch.bincount(filtered, minlength=num_classes)
                add_random = add_random/n_rows_unknown
                # Then combine the distribution + 'unknown' samples
                sizes = (((1-sampler.unknown_frac)*dist) +
                         (sampler.unknown_frac*add_random))
                sizes = sizes.to(utils.DEVICE)
                loss = loss_function(weight=1/sizes, # and take reciprocal
                                     ignore_index=utils.UNKNOWN_INT)

            loss.weighted = True
            loss.sampler_correction = False if sampler is None else True    
            losses.insert(0, loss)

        return losses

    def weighted_sampler(self, level='species', unknown_frac=0.5):
        '''Random sampler ensuring perfect class balance at specified level.
        Samples from unidentified class an `unknown_frac` fraction of times.'''
        
        lvl = utils.LEVELS.index(level) # Get index of level
        labels = self.taxonomies[:,lvl] # Get labels at level
        filtered = labels[labels != utils.UNKNOWN_INT] # Filter out unknowns
        num_classes = len(self.tax_encoder.lvl_encoders[lvl].classes_)
        non_zero = len(filtered.unique()) # Number of classes with >=1 sample

        # Calculating weights per class
        class_weights = ((1-unknown_frac)/
                (torch.bincount(filtered, minlength=num_classes)*(non_zero)))
        # Ensuring an unknown_frac proportion of unknown samples
        if len(self.taxonomies) != len(filtered): 
            sample_weights=((unknown_frac/(len(self.taxonomies)-len(filtered)))*
                                               torch.ones(len(self.taxonomies)))
        else: # Correction if there are no unknown samples
            class_weights += (unknown_frac/(1-unknown_frac))*class_weights
            sample_weights=torch.zeros(len(self.taxonomies)) 
        # Assigning weights per sample
        for i, entry in enumerate(self.taxonomies[:,lvl]):
            if entry != utils.UNKNOWN_INT:
                sample_weights[i] = class_weights[entry]
        # print(sample_weights.sum()) # NOTE uncomment to verify sum(weights)=1

        n_samples = min(len(self.taxonomies), utils.MAX_PER_EPOCH)
        sampler = tud.WeightedRandomSampler(sample_weights,n_samples)
        sampler.lvl = lvl
        sampler.unknown_frac = unknown_frac
        return sampler


class DataPrep: 
    '''ITS data preprocessing and preparing for use by mycoai
    
    Attributes
    ----------
    data: pd.DataFrame
        Holds dataframe containing taxonomies and sequences, with columns
        'phylum', 'class', 'order', 'genus', 'family', 'species', and 'sequence'
    '''

    def __init__(self, filename, tax_parser='unite', allow_duplicates=False,
                 name=None):
        '''Loads data into DataPrep object, using parser adjusted to data format
        
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
        
        tax_parsers = {'unite':self.unite_parser}
        if type(tax_parser) == str:
            tax_parser = tax_parsers[tax_parser]
        data = self.read_fasta(filename, tax_parser=tax_parser)

        if not allow_duplicates:
            if tax_parser is not None:
                data.drop_duplicates(subset=['sequence','species'],inplace=True)
            else:
                data.drop_duplicates(subset=['sequence'], inplace=True)

        self.data = data
        if name is None:
            self.name = utils.filename_from_path(filename)

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
                       valid_split=0.0, export_path=None):
        '''Converts data into a mycoai Dataset object
        
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
            else:
                dna_encoder = dna_encs[dna_encoder]() 
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
            data = self._encode_subset(self.data, dna_encoder, tax_encoder, self.name)
    
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
            sequences.append(dna_encoder.encode(row))
            taxonomies.append(tax_encoder.encode(row))

        if type(tax_encoder) == encoders.TaxonEncoder and tax_encoder.train:
            tax_encoder.finish_training()

        # Convert to tensors and store
        sequences = torch.stack(sequences)
        taxonomies = torch.stack(taxonomies)
        data = Dataset(sequences, taxonomies, dna_encoder, tax_encoder, name)

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
        '''Removes sequences with more than tolerated std from mean length'''
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