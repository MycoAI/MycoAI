'''Pre-training tasks for ITS transformer models.'''

import torch
import torch.utils.data as tud
from tqdm import tqdm
import wandb
from mycoai import utils
from mycoai.deep import train

class MLMTrainer:
    '''Masked Language Modelling: training a network to predict the value of
    randomly selected masked out input tokens.'''

    @staticmethod
    def train(model, data, epochs=100, batch_size=64, p_mlm=0.15, p_mask=0.8, 
              p_random=0.1, sampler=None, optimizer=None, warmup_steps=4000,
              label_smoothing=0.1, wandb_config={}, wandb_name=None):
        '''Trains an ITS transformer model on Masked Language Modelling task.
        
        Parameters
        ----------
        model: torch.nn.Module
            The to-be-trained model. Must be part of type BERT. 
        data: mycoai.TensorData
            TensorData object containing sequences to be used for training.
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
        warmup_steps: int | NoneType
            When specified, will use a learning rate schedule in which the lr 
            increases linearly for the first warmup_steps, then decreases 
            proportionally to 1/sqrt(step_number) (default is 4000)
        label_smoothing: float
            Adds uncertainty to target labels as regularization (default is 0.1)
        wandb_config: dict
            Extra information to be added to weights and biases config data.
        wandb_name: str
            Name of the run to be displayed on weights and biases.'''

        # PARAMETER INITIALIZATION
        # Data and sampling
        if sampler is None: # Random sampling as default
            sampler = torch.utils.data.RandomSampler(data)
        dataloader = tud.DataLoader(data, batch_size=batch_size,sampler=sampler)

        # Loss and optimizer
        loss_function = torch.nn.CrossEntropyLoss( # Ignore padding
            label_smoothing=label_smoothing, ignore_index=utils.TOKENS['PAD'])
        if optimizer is None: # Use Adam as default
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1, 
                                         betas=(0.9,0.98))
        if warmup_steps is None: # Constant learning rate as default
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lambda step: 0.0001) 
        else: # Initialize lr scheduler if warmup_steps is specified
            schedule = train.LrSchedule(model.d_model, warmup_steps)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                                  optimizer, lambda step: schedule.get_lr(step))
            
        # Mixed precision
        prec = torch.float16 if utils.DEVICE.type == 'cuda' else torch.bfloat16
        if utils.MIXED_PRECISION:
            scaler = torch.cuda.amp.grad_scaler.GradScaler()
        else:
            scaler = train.DummyScaler() # Does nothing 

        # Other configurations
        model.set_mode('mlm') # Turns on MLM layer of BERT model
        wandb_run = MLMTrainer.wandb_init(data, model, optimizer, sampler,
            loss_function, batch_size, epochs, p_mlm, p_mask, p_random, 
            warmup_steps, label_smoothing, wandb_config, wandb_name)
        wandb_run.watch(model, log='all')

        # TRAINING LOOP
        print("Training MLM task...") if utils.VERBOSE > 0 else None
        for epoch in tqdm(range(epochs)):
            model.train()
            epoch_loss, epoch_corr, epoch_seen = 0, 0, 0
            for x, _ in dataloader: # Loop through the data

                # Learning:
                # Apply mask to batch using the probability arguments
                x, y = MLMTrainer.mask_batch(x, model.vocab_size, 
                                             p_mlm, p_mask, p_random)
                x, y = x.to(utils.DEVICE), y.to(utils.DEVICE)
                optimizer.zero_grad()
                with torch.autocast(device_type=utils.DEVICE.type, dtype=prec,
                                    enabled=utils.MIXED_PRECISION):
                    y_pred = model(x) # Make a prediction
                    y = y.view(-1) # Flatten labels
                    y_pred = y_pred.view(-1, model.vocab_size) # Flatten 
                    loss = loss_function(y_pred, y) # Calculate loss
                scaler.scale(loss).backward() # Calculate gradients
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer) # Apply optimizer 
                lr_scheduler.step() # Update learning rate
                scaler.update() 

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

        # WRAPUP
        # Finishing the wandb_run, getting results dataframe
        wandb_run.finish(quiet=True)
        wandb_api = wandb.Api()
        wandb_id = f'{wandb_run.project}/{wandb_run._run_id}'
        model.pretraining = wandb_id
        run = wandb_api.run(wandb_id)
        history = run.history(pandas=True)
        
        return model, history

    @staticmethod
    def mask_batch(x, vocab_size, p_mlm=0.15, p_mask=0.8, p_random=0.1):
        '''Maks a batch of sequence data for MLM'''

        # Calculate boolean tensors using selection probabilities
        select = ((torch.rand(x.shape) < p_mlm) & # Select for MLM
                    (x != utils.TOKENS['PAD']) & # Can't select...
                    (x != utils.TOKENS['SEP']) & # ... special tokens
                    (x != utils.TOKENS['CLS_P']) &
                    (x != utils.TOKENS['CLS_C']) &
                    (x != utils.TOKENS['CLS_O']) &
                    (x != utils.TOKENS['CLS_F']) &
                    (x != utils.TOKENS['CLS_G']) &
                    (x != utils.TOKENS['CLS_S'])) 
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
                   p_mlm, p_mask, p_random, warmup_steps, label_smoothing, 
                   wandb_config, wandb_name):
        '''Initializes wandb_run, writes config'''
        utils.wandb_cleanup()
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
            'warmup_steps': warmup_steps,
            'label_smoothing': label_smoothing,
            **wandb_config,
            **utils.get_config()
        }
        return wandb.init(project=utils.WANDB_PROJECT, config=config, 
                          name=wandb_name, dir=utils.OUTPUT_DIR)