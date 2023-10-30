'''Pre-training tasks for ITS transformer models.'''

import torch
import torch.utils.data as tud
from tqdm import tqdm
import wandb
from mycoai import utils

class MLMTask:
    '''Masked Language Modelling: training a network to predict the value of
    randomly selected masked out input tokens.'''

    @staticmethod
    def train(model, data, epochs=100, batch_size=64, p_mlm=0.15, p_mask=0.8, 
              p_random=0.1, sampler=None, optimizer=None, wandb_config={},
              wandb_name=None):
        '''Trains an ITS transformer model on Masked Language Modelling task.
        
        Parameters
        ----------
        model: torch.nn.Module
            The to-be-trained model. Must be part of type BERT. 
        data: mycoai.Dataset
            Dataset object containing sequences to be used for training.
        epochs: int
            Number of training epochs (default is 100).
        batch_size: int
            Number of samples per batch (default is 64).
        p_mlm: float
            Percentage of tokens selected for MLM (default is 0.15)
        p_mask: float
            Percentage of MLM-selected tokens that will be masked out (default 
            is 0.8)
        p_random: float
            Percentage of MLM-selected tokens that will be replaced by a random
            token (default is 0.1)
        sampler: torch.utils.data.Sampler
            Strategy to use for drawing data samples
        optimizer: torch.optim
            Optimization strategy (default is Adam)
        wandb_config: dict
            Extra information to be added to weights and biases config data.
        wandb_name: str
            Name of the run to be displayed on weights and biases.'''
        
        # Initializing parameters
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, 
                                        weight_decay=0.0001)
        if sampler is None:
            sampler = torch.utils.data.RandomSampler(data)
        loss_function = torch.nn.CrossEntropyLoss(
                                               ignore_index=utils.TOKENS['PAD'])
        dataloader = tud.DataLoader(data, batch_size=batch_size,sampler=sampler)
        model.set_mode('mlm') # Turns on MLM layer of BERT model
        wandb_run = MLMTask.wandb_init(data, model, optimizer, sampler, 
                        loss_function, batch_size, epochs, p_mlm, p_mask, 
                        p_random, wandb_config, wandb_name)

        # Training loop
        for epoch in tqdm(range(epochs)):
            model.train()
            epoch_loss, epoch_corr, epoch_seen = 0, 0, 0
            for x, _ in dataloader: # Loop through the data

                # Learning:
                # Apply mask to batch using the probability arguments
                x, y = MLMTask.mask_batch(x, model.vocab_size, 
                                          p_mlm, p_mask, p_random)
                x, y = x.to(utils.DEVICE), y.to(utils.DEVICE)
                y_pred = model(x) # Make a prediction
                y = y.view(-1) # Flatten labels
                y_pred = y_pred.view(-1, model.vocab_size) # Flatten prediction
                loss = loss_function(y_pred, y) # Calculate loss
                optimizer.zero_grad()
                loss.backward() # Calculate gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step() # Apply optimizer 
                
                # Metrics:
                epoch_loss += x.size(0)*loss.item() # Calculate running loss
                labels_pred = torch.argmax(y_pred, dim=1)
                selected = (y != utils.TOKENS['PAD'])
                epoch_corr += (labels_pred[selected]==y[selected]).sum().item()
                epoch_seen += selected.sum().item()

            # Logging metrics after epoch
            wandb_run.log({'epoch':     epoch,
                           'loss':      epoch_loss/len(data),
                           'accuracy':  epoch_corr/epoch_seen})

        # Finishing the wandb_run, getting results dataframe
        wandb_run.finish(quiet=True)
        wandb_api = wandb.Api()
        run = wandb_api.run(f'{wandb_run.project}/{wandb_run._run_id}')
        history = run.history(pandas=True)
        
        return model, history

    @staticmethod
    def mask_batch(x, vocab_size, p_mlm=0.15, p_mask=0.8, p_random=0.1):
        '''Maks a batch of sequence data for MLM'''

        # Calculate boolean tensors using selection probabilities
        select = ((torch.rand(x.shape) < p_mlm) & # Select for MLM
                    (x != utils.TOKENS['PAD']) & # Can't select...
                    (x != utils.TOKENS['SEP']) & # ... special tokens
                    (x != utils.TOKENS['CLS']))
        probs = torch.rand(x.shape)
        masked = select & (probs < p_mask)
        random = select & (probs >= p_mask) & (probs < p_mask + p_random)

        # Replace with masks/random tokens using the selection tensors
        y = x.clone() # Create a copy
        x[masked] = utils.TOKENS['MASK'] # Apply mask
        x[random] = torch.randint( # Apply random tokens
            len(utils.TOKENS), # Exclude special tokens 
            vocab_size,
            (torch.sum(random).item(),), 
            dtype=torch.long)
        # The rest for which select is True remains unchanged
        y[~select] =  utils.TOKENS['PAD'] # Pad those not selected 

        return x, y
    
    @staticmethod
    def wandb_init(data, model, optimizer, sampler, loss, batch_size, epochs, 
                   p_mlm, p_mask, p_random, wandb_config, wandb_name):
        config = {
            'task': 'mlm',
            **utils.get_config(data, prefix='trainset'),
            **utils.get_config(model),
            **utils.get_config(optimizer, prefix='opt'),
            **utils.get_config(sampler, 'sampler'),
            **utils.get_config(loss, 'loss'),
            'batch_size': batch_size, 
            'epochs': epochs,
            'p_mlm': p_mlm, 
            'p_mask': p_mask,
            'p_random': p_random,
            **wandb_config
        }
        return wandb.init(project='MycoAI ITSClassifier', config=config, 
                          name=wandb_name, dir=utils.OUTPUT_DIR)


class NSPTask:
    '''Next Sentence Prediction''' #TODO

    def __init__(self):
        pass

    def train(self, data):
        pass