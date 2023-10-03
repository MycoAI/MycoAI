'''Contains data encoders that converts sequence/taxonomy strings to tensors.'''

import re
import torch
import sklearn
import itertools
import numpy as np
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

    def encode(self, data_row):
        raise RuntimeError('encode method not implemented for base class.')


class FourDimDNA(DNAEncoder):
    '''Encoding method for converting nucleotide bases into 4 channel arrays'''

    def __init__(self, length=1000):
        self.length = length

    def encode(self, data_row):
        '''Encodes a single data row, returns list of four-channel encodings'''

        encoding = ([IUPAC_ENCODING[data_row['sequence'][i]]
               for i in range(min(len(data_row['sequence']), self.length))])
        return encoding + [[0,0,0,0]]*(self.length-len(encoding)) # Padding
    

class KmerEncoder(DNAEncoder):
    '''Base clase for nucleotide encoders that are based on k-mers'''

    def __init__(self, k, alphabet): 
        self.words = [''.join(word) for word in itertools.product(alphabet+'?', 
                                                                  repeat=k)]
        self.alphabet = alphabet
        self.k = k 
        
    def _seq_preprocess(self, sequence):
        '''Replaces uncertain nucleotides with "?", cuts seq. to fit k-mers'''
        seq = re.sub('[^' + self.alphabet + ']', '?', sequence)
        seq = seq[:int(len(seq)/self.k)*self.k]
        return seq
    

class KmerTokenizer(KmerEncoder):
    '''K-mer DNA encoding method that represents a k-mer as an index/token'''

    def __init__(self, k=4, alphabet='ACGT', length=512):
        super().__init__(k, alphabet)
        self.length = length
        min_token = len(utils.TOKENS)
        self.map = {word:i+min_token for i, word in enumerate(self.words)}

    def encode(self, data_row):
        '''Encodes data row, returns list of (kmer-based) one-hot encodings'''
        encoding = [utils.TOKENS['CLS']]
        seq = self._seq_preprocess(data_row['sequence'])
        for i in range(0,min((self.length-2)*self.k, len(seq)),self.k):
            encoding.append(self.map[seq[i:i+self.k]]) # Add encoding
        encoding.append(utils.TOKENS['SEP']) # Add separator token 
        encoding += (self.length-len(encoding))*[utils.TOKENS['PAD']] # Padding
        return encoding


class KmerOneHot(KmerEncoder):
    '''K-mer DNA encoding method that represents a k-mer as one-hot vector'''

    def __init__(self, k=3, alphabet='ACGT', length=512): 
        super().__init__(k, alphabet)
        self.length = length
        min_token = len(utils.TOKENS)
        self.map = {word:i+min_token for i, word in enumerate(self.words)}
        self.num_tokens = len(self.map) + min_token

    def encode(self, data_row):
        '''Encodes data row, returns list of (kmer-based) one-hot encodings'''
        encoding = [self._get_onehot_vector(utils.TOKENS['CLS'])] 
        seq = self._seq_preprocess(data_row['sequence'])
        for i in range(0,min((self.length-2)*self.k, len(seq)),self.k):
            encoding.append(self._get_onehot_vector(self.map[seq[i:i+self.k]]))
        encoding.append(self._get_onehot_vector(utils.TOKENS['SEP']))
        encoding += (self.length-len(encoding))*[[0]*self.num_tokens]
        return encoding

    def _get_onehot_vector(self, index):
        empty = [0]*(self.num_tokens)
        empty[index] = 1
        return empty
    

class KmerSpectrum(KmerEncoder):
    '''Encoding method, converts each sequence to a k-mer frequency spectrum'''

    def __init__(self, k=4, alphabet='ACGT', normalize=True): 
        super().__init__(k, alphabet)
        self.map = {word:i for i, word in enumerate(self.words)}
        self.normalize = normalize

    def encode(self, data_row):
        '''Encodes a single data row, returns k-mer frequences'''
        # Replace characters not in alphabet with unknown marking
        seq = re.sub('[^' + self.alphabet + ']', '?', data_row['sequence'])
        freqs = np.zeros(len(self.map)) # Initialize frequency vector
        for i in range(len(seq)-self.k+1): # Count
            freqs[self.map[seq[i:i+self.k]]] += 1
        if self.normalize: # Normalize
            freqs = freqs/freqs.sum()
        return [list(freqs)] 


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
    
    def decode(self, labels: np.ndarray):
        '''Decodes an array of index labels into their corresponding strings'''
        decoding = []
        for i, encoder in enumerate(self.lvl_encoders):
            decoding.append(encoder.inverse_transform(labels[:,i]))
        decoding = np.stack(decoding,axis=1)
        return decoding
    
    def finish_training(self):
        '''Normalizes rows of inference matrices to obtain transition 
        probabilities of going from child class to parent class.'''
        for matrix in self.inference_matrices:
            for i in range(matrix.shape[0]):
                matrix[i] = matrix[i]/matrix[i].sum()
        self.train = False