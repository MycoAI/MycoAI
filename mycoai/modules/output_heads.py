'''Various types of output heads for taxon classification. Output is list of 
tensors (one for each level).'''

import torch
from mycoai import utils


class InferParent(torch.nn.Module):
    '''Infers parent classes by looking in the inference matrix and seeing what
    parent a child class is most often part of.'''

    def __init__(self, classes, tax_encoder, max_level='species'):
        super().__init__()
        self.tax_encoder = tax_encoder
        self.max_level = utils.LEVELS.index(max_level)
        self.fc1 = torch.nn.LazyLinear(out_features=classes[self.max_level])
        self.softmax = torch.nn.Softmax(dim=1) 
        self.classes = classes

    def forward(self, x):
        
        # Initialize output list with zeros until base level
        output = [torch.zeros((x.shape[0], self.classes[lvl]), device=x.device)
                                         for lvl in range(5,self.max_level,-1)]
        
        # At base level, make a prediction
        x = self.fc1(x) # Linear layer
        output.insert(0, self.softmax(x)) # Softmax
        _, pred = torch.max(output[0],1) # Get argmax

        # Above base level, infer parents 
        for i in range(self.max_level-1,-1,-1): 
            # Get parent indices by retrieving the argmax of inference matrix
            _, pred = torch.max(self.tax_encoder.inference_matrices[i][pred], 1)
            # Mimick probability tensor by one-hot encoding
            probs = torch.zeros((pred.shape[0], self.classes[i]))
            probs[torch.arange(pred.shape[0]), pred] = 1
            output.insert(0, probs.to(x.device))

        return output


class InferSum(torch.nn.Module):
    '''Sums softmax probabilities of child classes to infer the probability of 
    a parent, using inference matrices.'''

    def __init__(self, classes, tax_encoder, max_level='species'):
        super().__init__()
        self.tax_encoder = tax_encoder
        self.max_level = utils.LEVELS.index(max_level)
        self.fc1 = torch.nn.LazyLinear(out_features=classes[self.max_level])
        self.softmax = torch.nn.Softmax(dim=1) 
        self.classes = classes
    
    def forward(self, x):
        
        # Initialize output list with zeros until base level
        output = [torch.zeros((x.shape[0], self.classes[lvl]), device=x.device)
                                         for lvl in range(5,self.max_level,-1)]
        
        # At base level, make a prediction
        x = self.fc1(x) # Linear layer
        output.insert(0, self.softmax(x)) # Softmax

        # Above base level, infer parents
        for i in range(self.max_level-1,-1,-1):
            output.insert(0, self.tax_encoder.infer_parent_probs(output[0], i))

        return output    
    

class MultiHead(torch.nn.Module):
    '''Predicting multiple taxon levels using different heads.'''

    def __init__(self, classes, max_level='species'):
        super().__init__()
        self.max_level = utils.LEVELS.index(max_level) 
        self.output = torch.nn.ModuleList(
            [torch.nn.LazyLinear(out_features=classes[i]) 
             for i in range(self.max_level+1)])
        self.softmax = torch.nn.ModuleList(
            [torch.nn.Softmax(dim=1) for i in range(self.max_level+1)])
        self.classes = classes
    
    def forward(self, x, levels=range(6)):
        outputs = []
        for lvl in levels:
            if lvl > self.max_level: # Zeros if above max_level
                outputs.append(torch.zeros((x.shape[0], self.classes[lvl]), 
                                           device=x.device))
            else:
                outputs.append(self.softmax[lvl](self.output[lvl](x)))

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
        outputs.append(self.softmax[0](prev))
        for i in range(1, len(self.output)):
            prev = self.output[i](prev)
            outputs.append(self.softmax[i](prev))
        return outputs
    
    def _forward_use_prob(self, x):
        outputs = []
        prev = self.softmax[0](self.output[0](x))
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
    
    
class SoftmaxTreeNode(torch.nn.Module):
    ''''Node of a hierarchical softmax tree.'''

    def __init__(self, parent, childs, root, lvl):
        super().__init__()      
        
        # Finding subtrees
        if lvl < 5: # Species-level:
            subtrees = [SoftmaxTreeNode(child, root.find_children(child,lvl),
                        root, lvl+1) for child in childs]
            root.subtrees[lvl].append(torch.nn.ModuleList(subtrees)) 
        
        # Setting attributes
        self.generator = utils.Generator(root.input_dim, len(childs))
        self.parent = parent # Save class indices 
        self.child_encodings = childs # (= TaxonEncoder encodings)

    def forward(self, x): 
        return self.generator(x)
    

class SoftmaxTree(SoftmaxTreeNode):
    '''Root node of a hierarchical softmax tree. Output per class is the
    conditional probability given the parent, multiplied by the probability of
    the parent.'''

    def __init__(self, classes, tax_encoder, input_dim):
        self.tax_encoder = tax_encoder
        self.classes = classes
        self.input_dim = input_dim
        self.subtrees = [[] for i in range(5)]
        # Root is special instance of SoftmaxTreeNode
        super().__init__(None, torch.arange(classes[0]), self, 0)
        # Convert subtrees to module lists
        for i in range(5):
            self.subtrees[i] = torch.nn.ModuleList(self.subtrees[i])
        self.subtrees = torch.nn.ModuleList(self.subtrees)

    def find_children(self, parent, lvl):
        inference_matrix = self.tax_encoder.inference_matrices[lvl]
        max_probs = torch.argmax(inference_matrix, dim=1)
        adj_matrix = torch.zeros(inference_matrix.shape)
        adj_matrix[torch.arange(inference_matrix.shape[0]), max_probs] = 1
        return torch.nonzero(adj_matrix[:,parent]).squeeze(1)

    def forward(self, x):
        full_output = [self.generator(x)]
        for lvl in range(1,6):
            output = torch.zeros(x.shape[0], self.classes[lvl], 
                                 device=utils.DEVICE)
            for parent in self.subtrees[lvl-1]:
                for subtree in parent:
                    prob = full_output[lvl-1][:,subtree.parent].view(-1,1)
                    output[:,subtree.child_encodings] = prob * subtree(x)  
            full_output.append(output)
        return full_output