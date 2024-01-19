'''Implementation for hierarchical label smoothing'''

import torch 
from mycoai import utils

class LabelSmoothing:
    '''Transforms target labels into probability distributions in which the mass
    is smeared out over related classes (instead of one-hot encoding). 
    
    Relationships are determined by the hierarchy. E.g., at species-level, 
    species that most often occur in the same genus, family, order, etc. as the
    target label will receive the most weight.'''

    def __init__(self, tax_encoder, smoothing=[0.01]*6):
        '''Initializes LabelSmoothing object. 
        
        Parameters
        ----------
        tax_encoder: mycoai.data.encoders.TaxonEncoder
            Encoder that was used to generate the target labels. Its inference
            matrices are used for the hierarchical relationships.
        smoothing: list[float] 
            Describes the contribution of label smoothing per level. E.g. if the
            first value of this list is 0.1, a 0.1 amount of weight is 
            distributed over the subclasses that are part of this phylum. The 
            value at the last (6th) index corresponds to epsilon in standard
            label smoothing (i.e. no hierarchical information used). Sum must 
            not exceed 1 (default is [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])'''
        
        self.smoothing = torch.tensor(smoothing, dtype=torch.float32)
        self.tax_encoder = tax_encoder

    def __call__(self, target_labels):
        '''Transforms target labels into probability distributions in which the 
        mass is smeared out over related classes (instead of one-hot).'''

        # Initializing new target
        batch_size = target_labels.shape[0]
        new_target = [ # Divide over classes
            (self.smoothing[5]/(self.tax_encoder.classes[lvl]-1))* 
            torch.ones((batch_size, self.tax_encoder.classes[lvl]), 
                       device=utils.DEVICE)
            for lvl in range(6)
        ]

        lvl_probs = []
        for lvl in range(6):

            # See for which batch entries we actually have information
            known = target_labels[:,lvl] != utils.UNKNOWN_INT

            # Set the original target index
            new_target[lvl][known, target_labels[known, lvl]] = (
                                  1-self.smoothing[5]-sum(self.smoothing[:lvl]))

            # Add smoothing contribution
            for i in range(len(lvl_probs)): 

                # Update to match current level
                if i < len(lvl_probs)-1:
                    lvl_probs[i] = (
                        self.tax_encoder.infer_child_probs(lvl_probs[i], lvl)
                    )

                # When label is unknown (all nan), divide equally
                lvl_probs[i] = torch.nan_to_num(lvl_probs[i], 
                                                1 / lvl_probs[i].shape[-1])

                # Add to target
                new_target[lvl] += self.smoothing[i] * lvl_probs[i]

            if lvl < 5:
                # Get probability dist at deeper level (inference matrix column)
                probs = torch.zeros(
                    (target_labels.shape[0], self.tax_encoder.classes[lvl+1]), 
                    device=utils.DEVICE
                )
                probs[known] = ( # Extract column to get subclass counts ...
                    self.tax_encoder.inference_matrices[lvl]
                        [:,target_labels[known,lvl]].t()
                ) # ... and divide by sum to get probabilities
                probs = probs / probs.sum(dim=1, keepdim=True) 
                lvl_probs.append(probs)

        return new_target

    def get_config(self):
        return {'label_smoothing': self.smoothing}