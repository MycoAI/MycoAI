'''Contains the ITSClassifier class for complete ITS classification models.'''

import torch
import numpy as np
from .. import utils
from .output_heads import SingleHead, MultiHead, ChainedMultiHead, Inference

class ITSClassifier(torch.nn.Module): 
    '''Fungal taxonomic classification model based on ITS sequences. 
    Supports several architecture variations.'''

    def __init__(self, base_arch, dna_encoder, tax_encoder, fcn_layers=[], 
                 output='inference', target_levels=utils.LEVELS, dropout=0):
        '''Creates network based on specified archticture and encoders

        Parameters
        ----------
        base_arch: Architecture subclass
            The body for the neural network
        dna_encoder: DNAEncoder
            The DNA encoder used for the expected input
        tax_encoder: TaxonEncoder
            The label encoder used for the (predicted) labels
        fcn_layers: list[int]
            List of node numbers for fully connected part before the output head
        output: ['single'|'multi'|'chained'|'inference']
            The type of output head(s) for the neural network
        target_levels: list[str]
            Names of the taxon levels for the prediction tasks
        dropout: float
            Dropout percentage for the dropout layer
        '''
        super().__init__()
        
        self.target_levels = self._get_target_level_indices(target_levels)
        self.dna_encoder = dna_encoder
        self.tax_encoder = tax_encoder 
        self.classes = torch.tensor(
            [len(self.tax_encoder.lvl_encoders[i].classes_) 
             for i in self.target_levels])
        self.base_arch = base_arch
        self.dropout = torch.nn.Dropout(dropout)
        
        # The fully connected part
        fcn = []
        for i in range(len(fcn_layers)):
            fcn.append(torch.nn.LazyLinear(fcn_layers[i]))
            fcn.append(torch.nn.ReLU())
        self.fcn = torch.nn.ModuleList(fcn)
        if len(fcn_layers) > 0:
            self.bottleneck_index = np.argmin(fcn_layers)
        
        match output:
            case 'single':
                self.output = SingleHead(self.classes)
            case 'multi':
                self.output = MultiHead(self.classes)
            case 'chained':
                self.output = ChainedMultiHead(self.classes)
            case 'inference':
                self.output = Inference(self.classes, self.tax_encoder)

        self.to(utils.DEVICE)

    def _get_target_level_indices(self, target_levels):
        levels_indices = []
        for level in target_levels: # Converting string input to correct indices
            levels_indices.append(utils.LEVELS.index(level))
        levels_indices.sort()
        return levels_indices

    def forward(self, x):
        x = self.base_arch(x) 
        for fcn_layer in self.fcn:
            x = fcn_layer(x)
        x = self.output(x)
        return x

    # TODO reimplement dimensionality reduction (use old mycoai version)
    def forward_until_bottleneck(self, x):
        x = self.base_arch(x)
        for i in range(0, ((self.bottleneck_index+1)*2)-1):
            x = self.fcn[i](x)
        return x

    def predict(self, data, return_as='tensor', return_labels=False):
        '''Returns predictions for entire dataset.
        
        data: mycoai.Dataset
            Deathcap Dataset object containing sequence and taxonomy Tensors
        return_as: ['tensor'|'dataframe']
            Whether to return the predictions as list of tensors (a tensor per
            level) or as a tabular pandas Dataframe.
        return_labels: bool
            Whether to include the true target labels in the return'''

        dataloader = torch.utils.data.DataLoader(data, shuffle=False,
                        batch_size=utils.PRED_BATCH_SIZE)
        predictions, labels = [[] for i in range(len(self.classes))], []
        
        with torch.no_grad():
            for (x,y) in dataloader:
                x, y = x.to(utils.DEVICE), y.to(utils.DEVICE)
                prediction = self(x)
                for i in range(len(self.classes)):
                    predictions[i].append(prediction[i])
                labels.append(y)
            predictions = [torch.cat(level) for level in predictions]
        
        if return_as == 'tensor':
            if return_labels:
                return predictions, torch.cat(labels)
            else:
                return predictions
        elif return_as == 'dataframe':
            pass # TODO decode predictions into strings