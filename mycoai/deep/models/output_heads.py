'''Various types of output heads for taxon classification. Output is list of 
tensors (one for each level).'''

import torch

class SingleHead(torch.nn.Module):
    '''Predicting a single taxon level.'''
    
    def __init__(self, classes):
        super().__init__()

        self.fc1 = torch.nn.LazyLinear(out_features=classes)
        self.softmax = torch.nn.Softmax(dim=1) 

    def forward(self, x):
        x = self.fc1(x)
        output = self.softmax(x)
        return [output]

class MultiHead(torch.nn.Module):
    '''Predicting multiple taxon levels using different heads.'''

    def __init__(self, classes):
        super().__init__()
        self.output = torch.nn.ModuleList(
            [torch.nn.LazyLinear(out_features=classes[i]) 
             for i in range(len(classes))])
        self.softmax = torch.nn.ModuleList(
            [torch.nn.Softmax(dim=1) for i in range(len(classes))]) 
    
    def forward(self, x):
        outputs = [self.softmax[i](self.output[i](x)) 
                   for i in range(len(self.output))]
        return outputs
    
class ChainedMultiHead(torch.nn.Module):
    '''Like MultiHead, but each taxon level also gets input from parent lvl.'''

    def __init__(self, classes, ascending=True, use_prob=True, all_access=True):
        '''Initializes ChainedMultiHead output head
        
        Parameters
        ----------
        classes: torch.Tensor 
            Indicates the class sizes of each taxonomic level
        ascending: bool
            If True, will chain the output heads from species- to phylum-level. 
            If False, chains the output heads from phylum- to species-level.
        use_prob: bool
            Whether or not to apply the softmax operation to the output tensor
            before passing it to the next taxonomic rank (default is True).
        all_access: bool
            If True, all taxonomic levels will get access to what is outputted 
            by the base architecture. If False, only the first taxonomic level
            will get this information (default is True).'''
        
        super().__init__()
        
        if ascending: # Species -> phylum
            self.output = torch.nn.ModuleList(
                [torch.nn.LazyLinear(out_features=classes[i]) 
                for i in range(len(classes)-1,-1,-1)])
        else: # Phylum -> species
            self.output = torch.nn.ModuleList(
                [torch.nn.LazyLinear(out_features=classes[i]) 
                for i in range(len(classes))])

        self.softmax = torch.nn.ModuleList(
            [torch.nn.Softmax(dim=1) for i in range(len(classes))])

        if use_prob: # Set correct forward method
            if all_access:
                self.forward = self._forward_use_prob_all_access
            else:
                self.forward = self._forward_use_prob
        else:
            if all_access:
                self.forward = self._forward_all_access
            else:
                self.forward = self._forward
        if ascending: # Enforce taxonomic level order
            self._unreversed_forward = self.forward
            self.forward = lambda x: self._unreversed_forward(x)[::-1]
    
    def _forward(self, x):
        outputs = []
        prev = (self.output[0](x))
        outputs.append(self.softmax[i](prev))
        for i in range(1, len(self.output)):
            prev = self.output[i](prev)
            outputs.append(self.softmax[i](prev))
        return outputs
    
    def _forward_use_prob(self, x):
        outputs = []
        prev = self.softmax[i](self.output[0](x))
        outputs.append(prev)
        for i in range(1, len(self.output)):
            prev = self.softmax[i](self.output[i](prev))
            outputs.append(prev)
        return outputs
    
    def _forward_all_access(self, x):
        outputs = []
        prev = torch.tensor([])
        prev = prev.to(x.device)
        for i in range(len(self.output)):
            prev = self.output[i](torch.concat((x, prev), 1))
            outputs.append(self.softmax[i](prev))
        return outputs
    
    def _forward_use_prob_all_access(self, x):
        outputs = []
        prev = torch.tensor([])
        prev = prev.to(x.device)
        for i in range(len(self.output)):
            prev = self.softmax[i](self.output[i](torch.concat((x, prev), 1)))
            outputs.append(prev)
        return outputs

class SumInference(torch.nn.Module):
    '''Sums softmax probabilities of child classes to infer the probability of 
    a parent, using inference matrices.'''

    def __init__(self, classes, tax_encoder):
        super().__init__()
        self.tax_encoder = tax_encoder
        self.fc1 = torch.nn.LazyLinear(out_features=classes[-1])
        self.softmax = torch.nn.Softmax(dim=1) 
        self.classes = classes
    
    def forward(self, x):
        x = self.fc1(x)
        output = [self.softmax(x)]
        for i in range(len(self.classes)-2,-1,-1):
            output.insert(0, 
                          output[0] @ self.tax_encoder.inference_matrices[i])

        return output
    
class ParentInference(torch.nn.Module):
    '''Infers parent classes by looking in the inference matrix and seeing what
    parent a child class is most often part of.'''

    def __init__(self, classes, tax_encoder):
        super().__init__()
        self.tax_encoder = tax_encoder
        self.fc1 = torch.nn.LazyLinear(out_features=classes[-1])
        self.softmax = torch.nn.Softmax(dim=1) 
        self.classes = classes

    def forward(self, x):
        x = self.fc1(x) # Linear layer
        output = [self.softmax(x)] # Softmax + initialize output list
        _, pred = torch.max(output[0],1) # Get argmax
        for i in range(len(self.classes)-2,-1,-1): # From low to high level
            # Get parent indices by retrieving the argmax of inference matrix
            _, pred = torch.max(self.tax_encoder.inference_matrices[i][pred], 1)
            # Mimick probability tensor by one-hot encoding
            probs = torch.zeros((pred.shape[0], self.classes[i]))
            probs[torch.arange(pred.shape[0]), pred] = 1
            output.insert(0, probs.to(x.device))
        return output

class TokenizedLevels(torch.nn.Module):
    '''Bases prediction on embeddings of CLS tokens that indicate taxon level'''

    def __init__(self, classes):
        super().__init__()
        self.output = torch.nn.ModuleList(
            [torch.nn.LazyLinear(out_features=classes[i]) 
             for i in range(len(classes))])
        self.softmax = torch.nn.ModuleList(
            [torch.nn.Softmax(dim=1) for i in range(len(classes))])
        self.classes = classes 

    def forward(self, x):
        outputs = []
        for i in range(len(self.output)):
            outputs.append(self.softmax[i](self.output[i](x[:,i,:])))
        return outputs