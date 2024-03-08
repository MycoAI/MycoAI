''''For training and testing deep learning models on ITS classification task.'''

import torch
import wandb
import numpy as np
import torch.utils.data as tud
from tqdm import tqdm
from mycoai import utils
from mycoai import train
from mycoai.train import weight_schedules as ws
from mycoai.train.label_smoothing import LabelSmoothing
from mycoai.train.loss import CrossEntropyLoss


mean = lambda tensor, weights: (tensor @ weights) / weights.sum()


class SeqClassTrainer:
    '''Multi-class classification (for multiple taxonomic levels)'''

    @staticmethod
    def train(model, train_data, valid_data=None, epochs=50, loss=None,
              batch_size=64, sampler=None, optimizer=None, 
              metrics=utils.EVAL_METRICS, levels=utils.LEVELS,
              warmup_steps=None, label_smoothing=[0.02,0.02,0.02,0.02,0.02,0], 
              wandb_config={}, wandb_name=None):
        '''Trains a neural network to classify ITS sequences
  
        Parameters
        ----------
        model: torch.nn.Module
            Neural network architecture
        train_data: mycoai.data.TensorData
            Preprocessed dataset containing ITS sequences for training
        valid_data: mycoai.data.TensorData
            Preprocessed dataset containing ITS sequences for validation   
        epochs: int
            Number of training iterations (default is 50)
        loss: list | function
            To-be-optimized loss function (or list of functions per level) 
            (default is CrossEntropyLoss)
        batch_size: int
            Number of training examples per optimization step (default is 64)
        sampler: torch.utils.data.Sampler
            Strategy to use for drawing data samples
        optimizer: torch.optim
            Optimization strategy (default is Adam)
        metrics: dict{str:function}
            Evaluation metrics to report during training, provided as dictionary
            with metric name as key and function as value (default is accuracy, 
            balanced acuracy, precision, recall, f1, and mcc).
        levels: list | mycoai.train.weight_schedules
            Specifies the levels that should be trained (and their weights).
            Can be a list of strings, e.g. ['genus', 'species]. Can also be a 
            list of floats, indicating the weight per level, e.g. [0,0,0,0,1,1].
            Can also be a MycoAI weight schedule object, e.g. 
            Constant([0,0,0,0,1,1]) (default is utils.LEVELS).
        warmup_steps: int | NoneType
            When specified, the lr increases linearly for the first warmup_steps 
            then decreases proportionally to 1/sqrt(step_number). Works only for
            models with d_model attribute (BERT/EncoderDecoder) (default is 0).
        label_smoothing: list[float]
            List of six decimals that controls how much label smoothing should 
            be added per taxonomic level. The sixth element of this list  refers
            to the amount of weight that is divided uniformly over all classes. 
            Hence, [0,0,0,0,0,0.1] corresponds to standard label smoothing with 
            epsilon=0.1, whereas [0.02,02,0.02,0.02,0.02,0] corresponds to 
            hierarchical label smoothing with epsilon=0.1 (see paper/docs)
            (default is [0.02,0.02,0.02,0.02,0.02,0]).
        wandb_config: dict
            Extra information to be added to weights and biases config data.
        wandb_name: str
            Name of the run to be displayed on weights and biases.'''
        
        # PARAMETER INITIALIZATION
        # Data and sampling
        if sampler is None:
            n_samples = min(len(train_data.taxonomies), utils.MAX_PER_EPOCH)
            sampler = tud.RandomSampler(train_data, num_samples=n_samples)
        train_dataloader = tud.DataLoader(train_data, batch_size=batch_size, 
                                          sampler=sampler)
        if label_smoothing is None:
            label_smoothing = [0,0,0,0,0,0]
        label_smoothing = LabelSmoothing(model.tax_encoder, label_smoothing)
        
        # Loss and optimizer                                  
        if loss is None:
            loss = [CrossEntropyLoss() for i in range(6)]
        if warmup_steps is None: # Constant learning rate as default 
            if optimizer is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, 
                                             weight_decay=0.0001)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                             lambda step: 1)
        else: # Initialize lr scheduler if warmup_steps is specified
            if optimizer is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=1, 
                                             weight_decay=0.0001)
            # NOTE If you specify an optimizer here, the lr will be weighted
            schedule = train.LrSchedule(model.d_model, warmup_steps) 
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                                  optimizer, lambda step: schedule.get_lr(step))
        if type(levels)==list:
            if type(levels[0])==str:
                levels = [float(lvl in levels) for lvl in utils.LEVELS]
            weight_schedule = ws.Constant(levels)
        else:
            weight_schedule = levels

        # Mixed precision
        prec = torch.float16 if utils.DEVICE.type == 'cuda' else torch.bfloat16
        if utils.MIXED_PRECISION and utils.DEVICE.type == 'cuda':
            scaler = torch.cuda.amp.grad_scaler.GradScaler()
        else:
            scaler = train.DummyScaler() # Does nothing 

        # Other configurations
        log_columns = SeqClassTrainer.wandb_log_columns(metrics, 
                                                       (valid_data is not None))
        wandb_run = SeqClassTrainer.wandb_init(train_data, valid_data, model, 
            optimizer, weight_schedule, sampler, loss, batch_size, epochs, 
            warmup_steps, label_smoothing, wandb_config, wandb_name)
        
        # TRAINING LOOP
        print("Training classification task...") if utils.VERBOSE > 0 else None
        for epoch in tqdm(range(epochs)):

            if epoch == 1: # Can't watch first epoch due to lazy layers
                wandb_run.watch(model, log='all')
            model.train()
            w = weight_schedule(epoch).to(utils.DEVICE)
            running_loss = np.zeros(6)
            
            for (x,y_i) in train_dataloader: 
                # Make a prediction
                x, y_i = x.to(utils.DEVICE), y_i.to(utils.DEVICE)
                y = label_smoothing(y_i)
                optimizer.zero_grad()
                with torch.autocast(device_type=utils.DEVICE.type, dtype=prec, 
                                    enabled=utils.MIXED_PRECISION):
                    y_pred = model(x)
                    losses = torch.cat(
                        [loss[lvl](y_pred[lvl],y[lvl],y_i[:,lvl]).reshape(1) for 
                        lvl in range(6)]
                    )
                    mean_loss = mean(losses, w)

                scaler.scale(mean_loss).backward() # Calculate gradients
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer) # Apply optimizer 
                lr_scheduler.step() # Update learning rate
                scaler.update()
                running_loss += x.size(0)*losses.detach().cpu().numpy()
                
            # Validation results
            if valid_data is not None:
                scores = SeqClassTrainer.validate(model, valid_data, metrics)
            else:
                scores = SeqClassTrainer.validate(model, train_data, metrics)
            scores = np.concatenate([[epoch+1], running_loss/len(train_data),
                                     scores])
            wandb_run.log({column: score 
                           for column, score in zip(log_columns, scores)})

        # Finishing the wandb_run, getting results dataframe
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        wandb_run.config.update({'num_params': params})
        if epochs > 1:
            wandb_run.unwatch(model)
        wandb_api = wandb.Api()
        wandb_id = f'{wandb_run.project}/{wandb_run._run_id}'
        model.train_ref = wandb_id
        run = wandb_api.run(wandb_id)
        history = run.history(pandas=True)
        SeqClassTrainer.wandb_learning_curves(wandb_run, history, metrics, 
                                             valid_data is not None)
        wandb_run.finish(quiet=True)
        
        if utils.VERBOSE > 0:
            print("Training finished, log saved to wandb (see above).")
            print("Final accuracy scores:\n---------------------")
            if valid_data is not None:
                SeqClassTrainer.final_report(history, 'valid')
            else:
                SeqClassTrainer.final_report(history, 'train')
        
        return model, history
    
    @staticmethod
    def validate(model, data, metrics, ignore_unknowns=True):
        '''Evaluates performance of model for the specified `metrics`.
        
        Parameters
        ----------
        model: SeqClassNetwork
            The to-be-validated neural network
        data: TensorData
            Test data
        metrics: dict{str:function}
            Evaluation metrics to report during testing
        ignore_unknowns:
            Ignore the unrecognized labels in the dataset (default is True)'''

        model.eval()
        with torch.no_grad():
            y_pred, y = model._predict(data, return_labels=True)
            results = []
            for m in metrics:
                for lvl in range(6):
                    y_lvl, y_pred_lvl = y[:,lvl], y_pred[lvl]
                    if ignore_unknowns:
                        mask = y[:,lvl] != utils.UNKNOWN_INT
                        y_lvl = y_lvl[mask]
                        y_pred_lvl = y_pred_lvl[mask]
                    results.append(metrics[m](y_lvl.cpu().numpy(), 
                                                y_pred_lvl.cpu().numpy()))
        
        cons = SeqClassTrainer.consistency(y_pred, model.tax_encoder)
        return np.array(results + cons)
        
    @staticmethod    
    def consistency(full_prediction, tax_encoder):
        '''Calculates the percentage of which predictions for a parent taxon are 
        consistent with child predictions, following the taxon hierarchy.'''

        consistencies = []
        n_rows = len(full_prediction[0])
        for i in range(5):
            this_lvl = full_prediction[i]
            next_lvl = full_prediction[i+1]
            m = tax_encoder.inference_matrices[i] # Get inference matrix
            # Normalize rows, obtain conditional probabilities per child
            m = (m / m.sum(dim=1, keepdim=True)).cpu().numpy()
            cons = np.sum([m[b,a] for a,b in zip(this_lvl, next_lvl)]) / n_rows
            consistencies.append(cons)

        return consistencies

    @staticmethod
    def wandb_log_columns(metrics, use_valid, consistency=True):
        '''Returns a list of column names for the wandb log'''
        columns = ['Epoch']
        columns += [f'loss|train|{lvl}' for lvl in utils.LEVELS]
        dataset = 'valid' if use_valid else 'train'
        columns +=  [f'{metric}|{dataset}|{lvl}' 
                        for metric in metrics for lvl in utils.LEVELS]
        if consistency:
            columns += ([f'Consistency|{dataset}|{pair}' 
                            for pair in ['P-C', 'C-O', 'O-F', 'F-G', 'G-S']])
        return columns

    @staticmethod
    def wandb_init(train_data, valid_data, model, optimizer, weight_schedule, 
                   sampler, loss, batch_size, epochs, warmup_steps, 
                   label_smoothing, wandb_config, wandb_name):
        '''Initializes wandb_run, writes config'''
        utils.wandb_cleanup()
        config = {
            'task': 'classification',
            **utils.get_config(train_data, prefix='trainset'),
            **utils.get_config(valid_data, prefix='validset'),
            **utils.get_config(model),
            **utils.get_config(optimizer, prefix='opt'),
            **utils.get_config(weight_schedule, 'weight_sched'),
            **utils.get_config(sampler, 'sampler'),
            **utils.get_config(loss[0], 'loss'),
            **utils.get_config(label_smoothing),
            'batch_size': batch_size, 
            'epochs': epochs,
            'warmup_steps': warmup_steps,
            **wandb_config,
            **utils.get_config()
        }
        return wandb.init(project=utils.WANDB_PROJECT, config=config, 
                          name=wandb_name, dir=utils.OUTPUT_DIR)
    
    @staticmethod
    def wandb_learning_curves(wandb_run, history, metrics, use_valid):
        '''Visualizes training history in custom charts on WandB'''
        
        history.replace('NaN', np.nan, inplace=True)
        columns = [f'loss|train|{lvl}' for lvl in utils.LEVELS]
        for metric in metrics:
            dataset = 'valid' if use_valid else 'train'
            columns += [f'{metric}|{dataset}|{lvl}' for lvl in utils.LEVELS]
            wandb_run.log({f"{metric} learning curve": wandb.plot.line_series(
              xs=history['Epoch'].values, 
              ys=history[columns].values.T,
              keys=[", ".join(column.split("|")[1:]) for column in columns],
              title=metric,
              xname='Epoch'
            )})

    @staticmethod
    def final_report(history, train_or_valid):
        print(train_or_valid.capitalize() + ":")
        report = history[[f'Accuracy|{train_or_valid}|{lvl}' for lvl in 
                          utils.LEVELS]]
        columns = dict(zip(report.columns, utils.LEVELS))
        report = report.tail(1)
        report.rename(columns=columns, inplace=True)
        report.reset_index(drop=True, inplace=True)
        print(report)