''''For training and testing deep learning models on ITS classification task.'''

import torch
import wandb
import numpy as np
import pandas as pd
import torch.utils.data as tud
from tqdm import tqdm
from mycoai import utils
from mycoai import plotter
from mycoai.deep import train
from mycoai.deep.train import weight_schedules as ws
from mycoai.deep.train.label_smoothing import LabelSmoothing
from mycoai.deep.train.loss import CrossEntropyLoss


mean = lambda tensor, weights: (tensor @ weights) / weights.sum()


class DeepITSTrainer:
    '''Multi-class classification (for multiple taxonomic levels)'''

    @staticmethod
    def train(model, train_data, valid_data=None, epochs=100, loss=None,
              batch_size=64, sampler=None, optimizer=None, 
              metrics=utils.EVAL_METRICS, levels=['species'], 
              p_teacher_forcing=0, warmup_steps=None, label_smoothing=None,
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
            Number of training iterations
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
        levels: list | mycoai.deep.train.weight_schedules
            Specifies the levels that should be trained (and their weights).
            Can be a list of strings, e.g. ['genus', 'species]. Can also be a 
            list of floats, indicating the weight per level, e.g. [0,0,0,0,1,1].
            Can also be a MycoAI weight schedule object, e.g. 
            Constant([0,0,0,0,1,1]) (default is ['species']).
        warmup_steps: int | NoneType
            When specified, the lr increases linearly for the first warmup_steps 
            then decreases proportionally to 1/sqrt(step_number). Works only for
            models with d_model attribute (BERT/EncoderDecoder) (default is 0).
        p_teacher_forcing: float
            Float between 0 and 1 that indicates the probability of teacher
            forcing for a batch. Teacher forcing inputs decoder with true 
            (masked) target labels instead of doing an autoregressive 
            prediction. Works only for EncoderDecoder model (default is False).
        label_smoothing: list[int]
            How much label smoothing should be added per taxonomic level. Per 
            level, adds a label_smoothing[level] amount of 'noise' to the
            species that are part of the target label at that level. Must sum up
            to 1 (default is [0,0,0,0,0,1]).
        wandb_config: dict
            Extra information to be added to weights and biases config data.
        wandb_name: str
            Name of the run to be displayed on weights and biases.'''
        
        # PARAMETER INITIALIZATION
        # Data and sampling
        if sampler is None:
            sampler = torch.utils.data.RandomSampler(train_data)
        train_dataloader = tud.DataLoader(train_data, batch_size=batch_size, 
                                          sampler=sampler)
        if label_smoothing is None:
            label_smoothing = [0,0,0,0,0,0]
        label_smoothing = LabelSmoothing(model.tax_encoder, label_smoothing)
        
        # Loss and optimizer                                  
        if loss is None:
            loss = [torch.nn.CrossEntropyLoss() 
                    for i in range(6)]
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
                weight_schedule = [float(lvl in levels) for lvl in utils.LEVELS]
            weight_schedule = ws.Constant(weight_schedule)
        else:
            weight_schedule = levels

        # Mixed precision
        prec = torch.float16 if utils.DEVICE.type == 'cuda' else torch.bfloat16
        if utils.MIXED_PRECISION and utils.DEVICE.type == 'cuda':
            scaler = torch.cuda.amp.grad_scaler.GradScaler()
        else:
            scaler = train.DummyScaler() # Does nothing 

        # Other configurations
        metrics = {'Loss': loss, **metrics}
        log_columns = DeepITSTrainer.wandb_log_columns(metrics, 
                                                       (valid_data is not None))
        wandb_run = DeepITSTrainer.wandb_init(train_data, valid_data, model, 
            optimizer, weight_schedule, sampler, loss, batch_size, epochs, 
            warmup_steps, label_smoothing, wandb_config, wandb_name)
        
        # TRAINING LOOP
        print("Training classification task...") if utils.VERBOSE > 0 else None
        for epoch in tqdm(range(epochs)):

            if epoch == 1: # Errors for epoch 0 due to torch.nn.Lazy modules
                wandb_run.watch(model, log='all')
            model.train()
            w = weight_schedule(epoch).to(utils.DEVICE)
            
            for (x,y_i) in train_dataloader:
                # Make a prediction
                x, y_i = x.to(utils.DEVICE), y_i.to(utils.DEVICE)
                y = label_smoothing(y_i)
                optimizer.zero_grad()
                teacher_forcing = np.random.binomial(1, p_teacher_forcing)
                with torch.autocast(device_type=utils.DEVICE.type, dtype=prec, 
                                    enabled=utils.MIXED_PRECISION):
                    if teacher_forcing:
                        y_pred = model(x, y) # TODO needs adjustment for label smoothing
                    else:
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
                
            # Validation results
            scores = DeepITSTrainer.validate(model, train_data, metrics,
                                             label_smoothing)
            scores = np.concatenate([[epoch+1], scores])
            if valid_data is not None:
                scores = np.concatenate([scores, DeepITSTrainer.validate(
                                  model, valid_data, metrics, label_smoothing)])
            wandb_run.log({column: score 
                           for column, score in zip(log_columns, scores)})

        # Finishing the wandb_run, getting results dataframe
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        wandb_run.config.update({'num_params': params})
        wandb_run.unwatch(model)
        wandb_api = wandb.Api()
        wandb_id = f'{wandb_run.project}/{wandb_run._run_id}'
        model.train_ref = wandb_id
        run = wandb_api.run(wandb_id)
        history = run.history(pandas=True)
        DeepITSTrainer.wandb_learning_curves(wandb_run, history, metrics, 
                                             valid_data is not None)
        wandb_run.finish(quiet=True)
        
        if utils.VERBOSE > 0:
            print("Training finished, log saved to wandb (see above).")
            print("Final accuracy scores:\n---------------------")
            DeepITSTrainer.final_report(history, 'train')
            if valid_data is not None:
                DeepITSTrainer.final_report(history, 'valid')
        
        return model, history
    
    @staticmethod
    def validate(model, data, metrics, label_smoothing, ignore_unknowns=True):
        '''Evaluates performance of model for the specified `metrics`.
        
        Parameters
        ----------
        model: DeepITSClassifier
            The to-be-validated neural network
        data: UniteData
            Test data
        metrics: dict{str:function}
            Evaluation metrics to report during testing
        label_smoothing: LabelSmoothing
            Label smoothing applied to the label (for loss validation)
        ignore_unknowns:
            Ignore the unrecognized labels in the dataset (default is True)'''

        model.eval()
        with torch.no_grad():
            y_pred, y_i = model._predict(data, return_labels=True)
            y = label_smoothing(y_i)
            results = []
            for m in metrics:
                for lvl in range(6):
                    y_lvl, y_i_lvl, y_pred_lvl = y[lvl], y_i[:,lvl], y_pred[lvl]
                    if ignore_unknowns:
                        mask = y_i[:,lvl] != utils.UNKNOWN_INT
                        y_lvl = y_lvl[mask]
                        y_i_lvl = y_i_lvl[mask]
                        y_pred_lvl = y_pred_lvl[mask]
                    if m == 'Loss':
                        results.append(
                             metrics[m][lvl](y_pred_lvl, y_lvl, y_i_lvl).item())
                    else:
                        argmax_y_pred = torch.argmax(y_pred_lvl, dim=1)
                        results.append(metrics[m](y_i_lvl.cpu().numpy(), 
                                                  argmax_y_pred.cpu().numpy()))
        
        cons = DeepITSTrainer.consistency(y_pred, model.tax_encoder)
        return np.array(results + cons)
        
    @staticmethod    
    def consistency(full_prediction, tax_encoder):
        '''Calculates the percentage of which predictions for a parent taxon are 
        consistent with child predictions, following the taxon hierarchy.'''

        consistencies = []
        n_rows = len(full_prediction[0])
        for i in range(5):
            this_lvl = torch.argmax(full_prediction[i], dim=1)
            next_lvl = torch.argmax(full_prediction[i+1], dim=1)
            cons = np.sum([tax_encoder.inference_matrices[i].cpu().numpy()[b,a] 
                           for a,b in zip(this_lvl, next_lvl)]) / n_rows
            consistencies.append(cons)

        return consistencies

    @staticmethod
    def results_init(metrics):
        '''Initializes an empty results dataframe with correct rows/columns'''
        columns += [f'{metric}|{lvl}' for metric in metrics 
                                                       for lvl in utils.LEVELS]
        columns += ([f'Consistency|{pair}' 
                            for pair in ['P-C', 'C-O', 'O-F', 'F-G', 'G-S']])
        return pd.DataFrame(columns=columns)

    @staticmethod
    def wandb_log_columns(metrics, use_valid, consistency=True):
        '''Returns a list of column names for the wandb log'''
        columns = ['Epoch']
        for dataset in ['train', 'valid'][:int(use_valid)+1]:
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
        for metric in metrics:
            columns = DeepITSTrainer.wandb_log_columns([metric], 
                                                       use_valid, False)
            wandb_run.log({f"{metric} learning curve": wandb.plot.line_series(
              xs=history['Epoch'].values, 
              ys=history[columns[1:]].values.T,
              keys=[", ".join(column.split("|")[1:]) for column in columns[1:]],
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