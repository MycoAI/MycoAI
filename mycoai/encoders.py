'''Contains data encoders that converts sequence/taxonomy strings to tensors.'''

import torch
from . import utils

IUPAC_ENCODING = {'A':[1,    0,    0,    0   ],
                  'C':[0,    1,    0,    0   ],
                  'G':[0,    0,    1,    0   ],
                  'T':[0,    0,    0,    1   ],
                  'R':[0.5,  0,    0.5,  0   ],
                  'Y':[0,    0.5,  0,    0.5 ],
                  'S':[0,    0.5,  0.5,  0   ],
                  'W':[0.5,  0,    0,    0.5 ],
                  'K':[0,    0,    0.5,  0.5 ], 
                  'M':[0.5,  0.5,  0,    0   ],
                  'B':[0,    0.33, 0.33, 0.33],
                  'D':[0.33, 0,    0.33, 0.33],
                  'H':[0.33, 0.33, 0,    0.33],
                  'V':[0.33, 0.33, 0.33, 0   ],
                  'N':[0.25, 0.25, 0.25, 0.25]}

class DNAEncoder:
    '''Base class for nucleotide encoders'''

    def __init__(self):
        pass

class FourDimDNA(DNAEncoder):
    '''Encoding method for converting nucleotide bases into 4 channel arrays'''

    def __init__(self, max_length=1000):
        self.max_length = max_length

    def encode(self, data_row):
        '''Encodes a single data row, returns list of four-channel encodings'''

        encoding = ([IUPAC_ENCODING[data_row['sequence'][i]]
        for i in range(min(len(data_row['sequence']), self.max_length))])
        return encoding + [[0,0,0,0]]*(self.max_length-len(encoding)) # Padding
    
    def decode(self):
        pass # TODO?


class TaxonEncoder:
    '''Sparse categorical encoding method of taxonomic labels on 6 levels, 
    Allowing inference of higher levels from a low-level prediction.
    
    Parameters
    ----------
    data: pd.DataFrame
        Containing the UniteData. Must be sorted (ascending order)!
        
    Attributes
    ----------
    labels: list[str]
        Translation table which label (index) corresponds to which taxon (value)
    inference_matrices: list[torch.Tensor]
        List of tensors to multiply lower-level Softmax probabilities with to 
        infer higher level (denotes which higher level a label is part of)
    taxon_encodings: list
        Contains encodings for all data rows for which self.encode is called
    current_tax: list[str]
        Keeps track of current taxon, to know when to increment label iterator
    current_lab: list[int]
        Keeps track of current label (per level). 
    '''

    def __init__(self, data):
        self.labels = self.initialize_labels(data) # Translation table
        self.inference_matrices = self.initialize_inference_matrices() # Empty
        # Iterators to keep track of label name and value
        self.current_tax = ['', '', '', '', '', ''] 
        self.current_lab = [-1, -1, -1, -1, -1, -1] 
        self.train = True

    def initialize_labels(self, data):
        '''Creates translation list for which label (index) corresponds to which 
        taxon (str), accounting for synonyms between different parent taxons.'''

        labels = [[] for i in range(6)]
        classes = data.groupby('phylum') # Yes, this is ugly but it is fastest
        for t1, g1 in classes: # For name, group in groups
            labels[0].append(t1) # Append name (label) to list
            for t2, g2 in g1.groupby('class'): # Group on deeper level
                labels[1].append(t2) # And repeat...
                for t3, g3 in g2.groupby('order'):
                    labels[2].append(t3)
                    for t4, g4 in g3.groupby('family'):
                        labels[3].append(t4)
                        for t5, g5 in g4.groupby('genus'):
                            labels[4].append(t5) # ... until species level
                            for t6, g6 in g5.groupby('species'): 
                                labels[5].append(t6)

        for i in range(len(labels)): # Remove the unidentified label
            labels[i] = ([label for label in labels[i] 
                          if label != utils.UNKNOWN_STR])

        return labels

    def initialize_inference_matrices(self):
        '''Set emtpy (zero) inference matrices with correct sizes'''

        inference_matrices = []
        num_parents = 0 
        for level in self.labels:
            num_classes = len(level)
            if num_parents > 0:
                inference_matrix = torch.zeros((num_classes, num_parents),
                                               dtype=torch.float32) # TODO see whether we can story this as binary (bool) dtype!
                inference_matrix.to_sparse()
                inference_matrix = inference_matrix.to(utils.DEVICE)
                inference_matrices.append(inference_matrix)
            num_parents = num_classes

        return inference_matrices

    def encode(self, data_row):
        '''Assigns integers to taxonomic level. When self.train==True: also 
        build inference matrices, assuming this function is called for all data 
        rows in alph. order, uses self.taxon/current_lab to keep track.'''

        encoding = []
        for i in range(6): # Loop through levels
            if self.train:
                if data_row[utils.LEVELS[i]] == utils.UNKNOWN_STR:
                    encoding.append(utils.UNKNOWN_INT) # = loss will ignore this
                    continue
                # Due to the alph. sorting, we can increment if we see new tax.
                if data_row[utils.LEVELS[i]] != self.current_tax[i]:
                    self.current_tax[i] = data_row[utils.LEVELS[i]] 
                    self.current_lab[i] += 1 
                # This is for debugging & testing purposes (proof that it works)
                # if self.labels[i][self.current_lab[i]] != self.current_tax[i]:
                #   raise RuntimeError("Inconsistency found in labels.")
                if i > 0: # Set inference matrix to 1 at current class & parent
                    self.inference_matrices[i-1][self.current_lab[i],
                                                 self.current_lab[i-1]] = 1
                encoding.append(self.current_lab[i])
            else:
                try: 
                    label = self.labels[i].index(data_row[utils.LEVELS[i]])
                except ValueError:
                    label = utils.UNKNOWN_INT
                encoding.append(label)
    
        return encoding

    def decode(self):
        pass # TODO?

