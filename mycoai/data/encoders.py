'''Contains data encoders that converts sequence/taxonomy strings to tensors.'''

import io
import re
import torch
import sklearn
import itertools
import numpy as np
import sentencepiece as spm
from mycoai import utils

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

    def encode(self, sequence):
        raise RuntimeError('encode method not implemented for base class.')


class FourDimDNA(DNAEncoder):
    '''Encoding method for converting nucleotide bases into 4 channel arrays'''

    def __init__(self, length=1000):
        self.length = length

    def encode(self, sequence):
        '''Encodes a single data row, returns list of four-channel encodings'''

        encoding = ([IUPAC_ENCODING[sequence[i]]
               for i in range(min(len(sequence), self.length))])
        encoding = encoding + [[0,0,0,0]]*(self.length-len(encoding)) # Padding
        return torch.tensor(encoding, dtype=torch.float32).transpose(1,0)

    def get_config(self):
        return {'type':       utils.get_type(self),
                'max_length': self.length}
    

class BytePairEncoder(DNAEncoder):
    '''Tokenizer based on appearance frequency of base combinations'''

    def __init__(self, data, length=256, vocab_size=768, n_max=500000):
        if utils.VERBOSE > 0:
            print('Initializing Byte Pair Encoder...')
        mem_stream = io.BytesIO() # In-memory byte stream to save model to
        sequences = data.data.sample(n=min(n_max,len(data.data)),replace=False)
        sequences = iter(sequences['sequence'].tolist()) # Sentence iterator
        spm.SentencePieceTrainer.train(sentence_iterator=sequences, 
                                       vocab_size=vocab_size,
                                       model_type='bpe',
                                       model_writer=mem_stream,
                                       bos_id = utils.TOKENS['CLS'],
                                       eos_id = utils.TOKENS['SEP'],
                                       pad_id = utils.TOKENS['PAD'],
                                       unk_id = utils.TOKENS['UNK'],
                                       control_symbols = ['MASK'],
                                       add_dummy_prefix = False,
                                       character_coverage=1.0)
        
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor(model_proto=mem_stream.getvalue())
        self.length = length

    def encode(self, sequence):
        '''Encodes a single data row using the BPE encoder'''
        seq = re.sub('[^ACTG]', '?', sequence)
        encoding = self.sp.encode(seq)[:self.length-2] # Leave room for CLS/PAD
        encoding = [utils.TOKENS['CLS']] + encoding + [utils.TOKENS['SEP']] 
        encoding += (self.length-len(encoding))*[utils.TOKENS['PAD']] # Padding
        return torch.tensor(encoding, dtype=torch.long)

    def encode_fast(self, data):
        '''Encodes an entire MycoAI data object using the BPE encoder'''
        seqs = data.data['sequence'].str.replace('[^ACTG]', '?', regex=True)
        seqs = self.sp.encode(seqs.tolist()) # BPE encoding
        seqs = [torch.tensor( # Add special tokens and convert to tensor
            [utils.TOKENS['CLS']] + seq[:self.length-2] + [utils.TOKENS['SEP']], 
            dtype=torch.long) for seq in seqs] 
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, # Stack and add padding
                            batch_first=True, padding_value=utils.TOKENS['PAD'])
        return seqs[:,:self.length]

    def get_config(self):
        return {'type':       utils.get_type(self),
                'max_length': self.length,
                'vocab_size': self.vocab_size}
    

class KmerEncoder(DNAEncoder):
    '''Base clase for nucleotide encoders that are based on k-mers'''

    def __init__(self, k, alphabet, overlapping): 
        self.k = k 
        self.alphabet = alphabet
        self.stride = 1 if overlapping else k
        self.words = [''.join(word) for word in itertools.product(alphabet+'?', 
                                                                  repeat=k)]
        
    def _seq_preprocess(self, sequence):
        '''Replaces uncertain nucleotides with "?", cuts seq. to fit k-mers'''
        seq = re.sub('[^' + self.alphabet + ']', '?', sequence)
        length = (int((len(seq) - self.k)/self.stride)*self.stride) + self.k
        seq = seq[:length]
        return seq
    

class KmerTokenizer(KmerEncoder):
    '''K-mer DNA encoding method that represents a k-mer as an index/token'''

    def __init__(self, k=4, alphabet='ACGT', length=512, overlapping=False):
        super().__init__(k, alphabet, overlapping)
        self.length = length
        min_token = max(utils.TOKENS.values()) + 1
        self.map = {word:i+min_token for i, word in enumerate(self.words)}
        self.vocab_size = len(self.map) + min_token

    def encode(self, sequence):
        '''Encodes data row, returns tensor of (kmer-based) token encodings'''
        encoding = [utils.TOKENS[f'CLS']]
        seq = self._seq_preprocess(sequence)
        i = 0
        while i + self.k <= len(seq) and len(encoding) < self.length-1:
            encoding.append(self.map[seq[i:i+self.k]]) # Add encoding
            i += self.stride
        encoding.append(utils.TOKENS['SEP']) # Add separator token 
        encoding += (self.length-len(encoding))*[utils.TOKENS['PAD']] # Padding
        return torch.tensor(encoding, dtype=torch.int)
    
    def get_config(self):
        return {'type':       utils.get_type(self),
                'k':          self.k,
                'stride':     self.stride,
                'max_length': self.length,
                'vocab_size': self.vocab_size}


class KmerOneHot(KmerEncoder):
    '''K-mer DNA encoding method that represents a k-mer as one-hot vector'''

    def __init__(self, k=3, alphabet='ACGT', length=512, overlapping=False): 
        super().__init__(k, alphabet, overlapping)
        self.length = length
        min_token = max(utils.TOKENS.values()) + 1
        self.map = {word:i+min_token for i, word in enumerate(self.words)}
        self.vocab_size = len(self.map) + min_token

    def encode(self, sequence):
        '''Encodes data row, returns tensor of (kmer-based) one-hot encodings'''
        encoding = [self._get_onehot_vector(utils.TOKENS[f'CLS'])] 
        seq = self._seq_preprocess(sequence)
        i = 0
        while i + self.k <= len(seq) and len(encoding) < self.length-1:
            encoding.append(self._get_onehot_vector(self.map[seq[i:i+self.k]]))
            i += self.stride
        encoding.append(self._get_onehot_vector(utils.TOKENS['SEP']))
        encoding += (self.length-len(encoding))*[[0]*self.vocab_size]
        return torch.tensor(encoding, dtype=torch.float32).transpose(1,0)

    def _get_onehot_vector(self, index):
        empty = [0]*(self.vocab_size)
        empty[index] = 1
        return empty

    def get_config(self):
        return {'type':       utils.get_type(self),
                'k':          self.k,
                'stride':     self.stride,
                'max_length': self.length,
                'vocab_size': self.vocab_size}
    

class KmerSpectrum(KmerEncoder):
    '''Encoding method, converts each sequence to a k-mer frequency spectrum'''

    def __init__(self, k=4, alphabet='ACGT', normalize=True, overlapping=True): 
        super().__init__(k, alphabet, overlapping)
        self.map = {word:i for i, word in enumerate(self.words)}
        self.normalize = normalize

    def encode(self, sequence):
        '''Encodes a single data row, returns k-mer frequences'''
        seq = self._seq_preprocess(sequence)
        freqs = np.zeros(len(self.map)) # Initialize frequency vector
        for i in range(0, len(seq)-self.k+1, self.stride): # Count
            freqs[self.map[seq[i:i+self.k]]] += 1
        if self.normalize: # Normalize
            freqs = freqs/(freqs.sum()+1e-7)
        return torch.tensor([list(freqs)], dtype=torch.float32)

    def get_config(self):
        return {'type':       utils.get_type(self),
                'k':          self.k,
                'stride':     self.stride,
                'normalize':  self.normalize} 


class TaxonEncoder:
    '''Sparse categorical encoding method of taxonomic labels on 6 levels, 
    Allowing inference of higher levels from a low-level prediction.
    
    Parameters
    ----------
    data: pd.DataFrame
        Containing the ITS data.
        
    Attributes
    ----------
    lvl_encoders: list[sklearn.preprocessing.LabelEncoder]
        A label encoder per level
    inference_matrices: list[torch.Tensor]
        List of tensors to multiply lower-level Softmax probabilities with to 
        infer higher level (denotes which higher level a label is part of)
    classes: list
        List with number of known classes per taxonomic level'''

    def __init__(self, data):
        self.lvl_encoders = self.initialize_labels(data.data)
        self.inference_matrices = self.initialize_inference_matrices() # Empty
        self.train = True

    def initialize_labels(self, data):
        '''Creates an encoder for each taxonomic level'''
        lvl_encoders = []
        for lvl in utils.LEVELS:
            encoder = sklearn.preprocessing.LabelEncoder()
            encoder.fit(data[data[lvl]!=utils.UNKNOWN_STR][lvl].unique())
            lvl_encoders.append(encoder)
        self.classes = [len(lvl_encoders[i].classes_) for i in range(6)]
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
        
        return torch.tensor(encoding, dtype=torch.int64)

    def encode_fast(self, data):
        '''Assigns integers to taxonomic level for entire MycoAI data object.'''

        # Init to utils.UKNOWN_INT
        data = data.data
        encoding = utils.UNKNOWN_INT*np.ones((len(data),6), dtype=int) 
        for i in range(6):
            # Filter for only known taxonomic labels
            known = data[utils.LEVELS[i]].isin(self.lvl_encoders[i].classes_)
            encoding[known, i] = ( # Encode
                self.lvl_encoders[i].transform(data[known][utils.LEVELS[i]])
            )

        return torch.tensor(encoding, dtype=torch.int64)
    
    def decode(self, labels: np.ndarray, levels: list=utils.LEVELS):
        '''Decodes an array of index labels into their corresponding strings'''
        decoding = []
        for i, l in enumerate(levels):
            l = utils.LEVELS.index(l)
            decoding.append(self.lvl_encoders[l].inverse_transform(labels[:,i]))
        decoding = np.stack(decoding,axis=1)
        return decoding
    
    def finish_training(self):
        for i in range(5):
            m = torch.zeros_like(self.inference_matrices[i])
            # Select most occurring parent per child
            parents = torch.argmax(self.inference_matrices[i], 1, True)
            m = m.scatter_(1, parents, 1)
            self.inference_matrices[i] = m
        self.train = False

    def infer_parent_probs(self, y, parent_lvl):
        '''Calculates probabilities for parents given child probabilities'''
        m = self.inference_matrices[parent_lvl] # Get inference matrix
        y = y.to(m.device)
        # Normalize rows, obtain conditional probabilities per child
        m = m / (m.sum(dim=1, keepdim=True) + 1e-7) 
        return y @ m

    def infer_child_probs(self, y, child_lvl):
        '''Calculates probabilities for children given parent probabilities'''
        m = self.inference_matrices[child_lvl-1] # Get inference matrix
        y = y.to(m.device)
        # Normalize columns, obtain conditional probabilities per parent
        m = m / (m.sum(dim=0) + 1e-7)
        return y @ m.t()

    def flat_label(self, y, lvl):
        '''Maps y from specified level to unique number for ALL levels.'''
        return torch.minimum(torch.full_like(y, sum(self.classes)), 
                             y + sum(self.classes[:lvl]))