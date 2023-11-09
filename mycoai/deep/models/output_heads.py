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

    def __init__(self, classes):
        super().__init__()
        self.output = torch.nn.ModuleList(
            [torch.nn.LazyLinear(out_features=classes[i]) 
             for i in range(len(classes))])
        self.softmax = torch.nn.ModuleList(
            [torch.nn.Softmax(dim=1) for i in range(len(classes))]) 
    
    def forward(self, x):
        outputs = []
        prev = torch.tensor([])
        prev = prev.to(x.device)
        for i in range(len(self.output)):
            prev = self.softmax[i](self.output[i](torch.concat((x, prev), 1)))
            outputs.append(prev)

        return outputs

class Inference(torch.nn.Module):
    '''Uses inference matrices to predict multiple target taxon levels, 
    while preserving their hierarchical structure.'''

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