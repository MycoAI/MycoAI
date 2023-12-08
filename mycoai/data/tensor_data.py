'''Contains neural-network-readable mycoai TensorData class'''

import torch
import pandas as pd
import torch.utils.data as tud
from mycoai import utils

class TensorData(torch.utils.data.Dataset):
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
        '''Initializes TensorData object from tensors / imports from filepath'''

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
        self.sequences = content['sequences'].to(utils.DEVICE)
        self.taxonomies = content['taxonomies'].to(utils.DEVICE)
        self.dna_encoder = content['dna_encoder']
        self.tax_encoder = content['tax_encoder']
        self.name = content['name']

        # Make sure inference matrices are on a used device
        inference_matrices = getattr(self.tax_encoder, 'inference_matrices', [])
        for i in range(len(inference_matrices)):
            inference_matrices[i] = inference_matrices[i].to(utils.DEVICE)
        if len(inference_matrices) > 0:
            self.tax_encoder.inference_matrices = inference_matrices

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
                # Check if we even have an unknown class at sampler level
                if n_rows_unknown == 0:
                    sampler.unknown_frac = 0
                    add_random = torch.zeros(add_random.shape)
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