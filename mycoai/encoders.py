'''Contains data encoders that converts sequence/taxonomy strings to tensors.'''

import torch
import sklearn
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
    lvl_encoders: list[sklearn.preprocessing.LabelEncoder]
        A label encoder per level
    inference_matrices: list[torch.Tensor]
        List of tensors to multiply lower-level Softmax probabilities with to 
        infer higher level (denotes which higher level a label is part of)'''

    def __init__(self, data):
        self.lvl_encoders = self.initialize_labels(data)
        self.inference_matrices = self.initialize_inference_matrices() # Empty
        self.train = True

    def initialize_labels(self, data):
        '''Creates an encoder for each taxonomic level'''
        lvl_encoders = []
        for lvl in utils.LEVELS:
            encoder = sklearn.preprocessing.LabelEncoder()
            encoder.fit(data[data[lvl]!=utils.UNKNOWN_STR][lvl].unique())
            lvl_encoders.append(encoder)
        return lvl_encoders 

    def initialize_inference_matrices(self):
        '''Set emtpy (zero) inference matrices with correct sizes'''

        inference_matrices = []
        num_parents = 0 
        for level in self.lvl_encoders:
            num_classes = len(level.classes_)
            if num_parents > 0: 
                inference_matrix = torch.zeros((num_classes, num_parents),
                                               dtype=torch.float32)
                inference_matrix.to_sparse()
                inference_matrix = inference_matrix.to(utils.DEVICE)
                inference_matrices.append(inference_matrix)
            num_parents = num_classes

        return inference_matrices
    
    def encode(self, data_row):
        '''Assigns integers to taxonomic level. When self.train==True: also 
        build inference matrices, assumes this method is called for all rows.'''

        encoding = [] # Init
        for i in range(6): # Loop through levels
            lvl = utils.LEVELS[i] # Label unidentified classes correctly
            if data_row[lvl] == utils.UNKNOWN_STR:
                encoding.append(utils.UNKNOWN_INT)
            else: # Use encoder to label known classes
                try:
                    encoding.append(self.lvl_encoders[i]
                                        .transform([data_row[lvl]])[0])
                except ValueError: # For classes in test data absent in training 
                    encoding.append(utils.UNKNOWN_INT)
            if self.train and i > 0: # For creating the inference matrices
                m = self.inference_matrices[i-1]
                parent, child = encoding[-2], encoding[-1]
                if parent != utils.UNKNOWN_INT and child != utils.UNKNOWN_INT:
                    m[child,parent] += 1 # Both known, add 1
                elif parent != utils.UNKNOWN_INT and child == utils.UNKNOWN_INT:
                    probs = 1/m.shape[0] # Only parent known, divide over childs
                    m[:,parent] += probs
                elif parent == utils.UNKNOWN_INT and child != utils.UNKNOWN_INT:
                    probs = 1/m.shape[1] # Only child known, divide over parents
                    m[child] += probs
                else: # Both unknown, divide probabilities over entire matrix
                    probs = 1 / (m.shape[0] * m.shape[1])
                    self.inference_matrices[i-1] += probs
        
        return encoding
    
    def decode(self, data_row):
        '''TODO'''

    def finish_training(self):
        '''Normalizes rows of inference matrices to obtain transition 
        probabilities of going from child class to parent class.'''
        for matrix in self.inference_matrices:
            for i in range(matrix.shape[0]):
                matrix[i] = matrix[i]/matrix[i].sum()
        self.train = False