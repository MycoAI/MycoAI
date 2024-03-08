'''Contains the SeqClassNetwork class for ITS classification models.'''

import time
import torch
import numpy as np
import pandas as pd
from mycoai import utils, data
import mycoai.modules.output_heads as mmo
from mycoai.modules.transformers import BERT, EncoderDecoder

class SeqClassNetwork(torch.nn.Module): 
    '''Performs taxonomic classification based on DNA sequences. It is a wrapper
    class that stores encoder objects and adds an output layer as well as other 
    application-related functionalities to a base architecture.'''

    def __init__(self, base_arch, dna_encoder, tax_encoder, fcn_layers=[], 
                 dropout=0, output='multi', max_level='species',
                 chained_config=[False,True,True]):
        '''Creates network based on specified archticture and encoders

        Parameters
        ----------
        base_arch: torch.nn.Module
            The body for the neural network
        dna_encoder: DNAEncoder
            The DNA encoder used for the expected input
        tax_encoder: TaxonEncoder
            The label encoder used for the (predicted) labels
        fcn_layers: list[int]
            List of node numbers for fully connected part before the output head
        dropout: float
            Dropout percentage for the dropout layer
        output:'infer_parent'|'infer_sum'|'multi'|'chained'|'tree'
            The type of output head(s) for the neural network.
        max_level: str
            Until what level to predict (only for 'infer_parent', 'infer_sum', 
            and 'multi' output heads, default is 'species').
        chained_config: list[bool]
            List of length 3 indicating the configuration for ChainedMultiHead.
            Corresponding to arguments: ascending, use_probs, and all_access.
            Default is [False, True, True].'''
        
        super().__init__()
        self.dna_encoder = dna_encoder
        self.tax_encoder = tax_encoder 
        self.classes = torch.tensor(
            [self.tax_encoder.classes[i] for i in range(6)])
        self.base_arch = base_arch
        self.dropout = torch.nn.Dropout(dropout)
        self.masked_levels = []
        
        if type(self.base_arch) == BERT:
            self.base_arch.set_mode('classification')
        if type(self.base_arch) == EncoderDecoder:
            self.forward = self._forward_encoder_decoder
            if output != 'multi':
                raise ValueError('EncoderDecoder base architecture only \
                                 supports multi-head output layer.')
        else:
            self.forward = self._forward
        d_hidden = getattr(self.base_arch, 'd_model', None)

        # The fully connected part
        fcn = []
        for i in range(len(fcn_layers)):
            fcn.append(torch.nn.LazyLinear(fcn_layers[i]))
            fcn.append(torch.nn.ReLU())
        self.fcn = torch.nn.ModuleList(fcn)
        if len(fcn_layers) > 0:
            self.bottleneck_index = np.argmin(fcn_layers)
        
        if output == 'infer_parent':
            self.output = mmo.InferParent(self.classes, tax_encoder, max_level)
            self.max_level = max_level
        elif output == 'infer_sum':
            self.output = mmo.InferSum(self.classes, tax_encoder, max_level)
            self.max_level = max_level
        elif output == 'multi':
            self.output = mmo.MultiHead(self.classes, max_level)
            self.max_level = max_level
        elif output == 'chained':
            self.output = mmo.ChainedMultiHead(self.classes, *chained_config)
            self.chained_config = chained_config
        elif output == 'tree':
            self.output = mmo.SoftmaxTree(self.classes, tax_encoder, d_hidden)
 
        self.to(utils.DEVICE)

    def _forward(self, x):
        x = self.base_arch(x) 
        for fcn_layer in self.fcn:
            x = fcn_layer(x)
        x = self.output(x)
        return x
    
    def _forward_encoder_decoder(self, x, y=None):
        if y is None:
            return self._forward_autoregressive(x)
        else:
            return self._forward_teacher_forcing(x, y)
    
    def _forward_autoregressive(self, x):
        '''Input decoder with prediction at previous token'''
        # Initialize target with CLS token (=0 for decoder)
        tgt = torch.zeros((x.shape[0],1), dtype=int, device=utils.DEVICE) 
        output = []
        for lvl in range(6):
            x_ = self.base_arch(x, tgt)[:,lvl]
            x_ = self.output(x_, [lvl])
            output += x_
            pred = 1+self.tax_encoder.flat_label(torch.argmax(x_[0],dim=-1),lvl)
            tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        return output

    def _forward_teacher_forcing(self, x, y):
        '''Input decoder with true target label'''
        # Initialize target with CLS token (=0 for decoder)
        tgt = torch.zeros((x.shape[0],1), dtype=int, device=utils.DEVICE) 
        output = []
        for lvl in range(6): 
            x_ = self.base_arch(x, tgt)[:,lvl]
            x_ = self.output(x_, [lvl])
            output += x_
            next = 1 + self.tax_encoder.flat_label(y[:,lvl], lvl)
            tgt = torch.cat((tgt, next.unsqueeze(1)), dim=1)
        return output

    def _forward_latent(self, x):
        '''Partial forward pass, stops at bottleneck to get latent space'''
        x = self.base_arch(x)
        if len(self.fcn) > 0:
            for i in range(0, ((self.bottleneck_index+1)*2)-1):
                x = self.fcn[i](x)
        return x     

    def classify(self, input_data):
        '''Classifies sequences in FASTA file, Data, or TensorData object, 
        returns a pandas DataFrame.'''

        t0 = time.time()
        self.eval()
        input_data, ids = self._encode_input_data(input_data, return_ids=True)
        predictions = self._predict(input_data)
        predictions = [pred_level.cpu().numpy() for pred_level in predictions]
        predictions = np.stack(predictions, axis=1)
        predictions = self.tax_encoder.decode(predictions)
        t1 = time.time()
        
        if utils.VERBOSE > 0:
            print(f'Classification took {t1-t0} seconds.')
        classification = np.concatenate((ids, predictions), 1)
        classification = pd.DataFrame(classification, 
                                      columns=['id'] + utils.LEVELS)
        classification[self.masked_levels] = utils.UNKNOWN_STR
        return classification
    
    def set_max_level(self, level):
        '''For multi-head output, changes the max_level.'''

        if type(self.output) != mmo.MultiHead:
            raise ValueError("Can only change max_level for multi head model.")
        self.output.max_level = utils.LEVELS.index(level)
        if self.output.max_level < 5:
            # Makes sure that levels deeper than max_level get masked out
            self.set_masked_levels(utils.LEVELS[self.output.max_level+1:])

    def set_masked_levels(self, levels: list):
        '''These levels will always return utils.UNKNOWN_STR in classification.
        Useful when model does not have a sufficient accuracy on a level.'''
        self.masked_levels = levels
    
    def _predict(self, data, return_labels=False):
        '''Returns predictions for entire dataset.
        
        data: mycoai.TensorData
            Deathcap TensorData object containing sequence and taxonomy Tensors
        return_labels: bool
            Whether to include the true target labels in the return'''

        dataloader = torch.utils.data.DataLoader(data, shuffle=False,
                        batch_size=utils.PRED_BATCH_SIZE)
        predictions, labels = [[] for i in range(len(self.classes))], []
        
        with torch.no_grad():
            for (x,y) in dataloader:
                x, y = x.to(utils.DEVICE), y.to(utils.DEVICE)
                y_pred = self(x)
                for i in range(len(self.classes)):
                    predictions[i].append(torch.argmax(y_pred[i], dim=1).cpu())
                labels.append(y.cpu())
            predictions = [torch.cat(level) for level in predictions]
        
        if return_labels:
                return predictions, torch.cat(labels)
        else:
            return predictions
        
    def _encode_input_data(self, input_data, return_ids=False):
        '''Encodes a FASTA file/Data object into TensorData object.'''

        ids = None
        if type(input_data) == str:
            input_data = data.Data(input_data, tax_parser=None, 
                                   allow_duplicates=True)
        if type(input_data) == data.Data:
            ids = input_data.data['id'].values
            input_data = input_data.encode_dataset(self.dna_encoder,
                                                   self.tax_encoder)
        if type(input_data) == data.TensorData:
            ids = np.array(['']*len(input_data)) if ids is None else ids
        else:
            raise ValueError("Input_data should be a FASTA filepath, " + 
                             "Data or TensorData object.")
        
        if return_ids:
            return input_data, np.expand_dims(ids, axis=1)
        else:
            return input_data
        
    def latent_space(self, input_data):
        '''Extracts latent space for given input data'''

        # Data processing
        self.eval()        
        input_data = self._encode_input_data(input_data)
        dataloader = torch.utils.data.DataLoader(input_data, shuffle=False,
                                               batch_size=utils.PRED_BATCH_SIZE)

        latent_repr = []
        with torch.no_grad():
            for (x,y) in dataloader: # Loop through batches
                x = x.to(utils.DEVICE)
                latent_repr.append(self._forward_latent(x).cpu()) # Forward pass
        return torch.cat(latent_repr).cpu().numpy() # Combine batches

    def multi_to_infer_sum(self):
        '''Replaces the MultiHead output with InferSum, using the species-level
        MultiHead output as a base.'''

        if type(self.output) != mmo.MultiHead:
            raise TypeError('self.output not of type MultiHead')

        new_head = mmo.InferSum(self.classes, self.tax_encoder, 'species')
        new_head.fc1 = self.output.output[5]
        new_head = new_head.to(utils.DEVICE)
        self.output = new_head

    def get_config(self):
        '''Returns configuration dictionary of this instance.'''
        dna_encoder = utils.get_config(self.dna_encoder, 'dna_encoder')
        base_arch = utils.get_config(self.base_arch, 'base_arch')
        dummy = [None, None, None]
        config = {
            'fcn': [self.fcn[i].out_features for i in range(0,len(self.fcn),2)],
            'output_type': utils.get_type(self.output),
            'output_chained_ascending':getattr(self,'chained_config', dummy)[0],
            'output_chained_use_probs':getattr(self,'chained_config', dummy)[1],
            'output_chained_allaccess':getattr(self,'chained_config', dummy)[2],
            'output_max_level': getattr(self, 'max_level', None),
            'train_ref': getattr(self, 'train_ref', None)
        }
        return {**dna_encoder, **base_arch, **config}
