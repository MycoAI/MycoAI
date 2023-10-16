''''For training and testing deep learning models on ITS classification task.'''

import time
import torch
import wandb
import numpy as np
import pandas as pd
import torch.utils.data as tud
import sklearn.metrics as skmetric
from functools import partial
from tqdm import tqdm
from mycoai import utils, plotter
from mycoai.training import weight_schedules as ws


EVAL_METRICS = {'Accuracy': skmetric.accuracy_score,
                'Accuracy (balanced)': skmetric.balanced_accuracy_score,
                'Precision': partial(skmetric.precision_score, 
                                     average='macro', zero_division=np.nan),
                'Recall': partial(skmetric.recall_score, 
                                  average='macro', zero_division=np.nan),
                'F1': partial(skmetric.f1_score, 
                              average='macro', zero_division=np.nan),
                'MCC': skmetric.matthews_corrcoef}

mean = lambda tensor, weights: (tensor @ weights) / weights.sum()


class ClassificationTask:
    '''Multi-task classification (for multiple taxonomic levels)'''

    @staticmethod
    def train(model, train_data, valid_data=None, epochs=100, loss=None,
              batch_size=64, sampler=None, optimizer=None, metrics=EVAL_METRICS, 
              weight_schedule=None, wandb_config={}, wandb_name=None):
        '''Trains a neural network to classify ITS sequences
  
        Parameters
        ----------
        model: torch.nn.Module
            Neural network architecture
        train_data: mycoai.data.Dataset
            Preprocessed dataset containing ITS sequences for training
        valid_data: mycoai.data.Dataset
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
        weight_schedule:
            Factors by which each level should be weighted in loss per epoch 
            (default is Constant([1,1,1,1,1,1]))
        wandb_config: dict
            Extra information to be added to weights and biases config data.
        wandb_name: str
            Name of the run to be displayed on weights and biases.'''
        
        # Initializing/setting parameter values
        if loss is None:
            loss = [torch.nn.CrossEntropyLoss(ignore_index=utils.UNKNOWN_INT) 
                    for i in range(6)]
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, 
                                        weight_decay=0.0001)
        if weight_schedule is None:
            weight_schedule = ws.Constant([1]*len(model.target_levels))
        train_dataloader = tud.DataLoader(train_data, batch_size=batch_size, 
                                          sampler=sampler)
        metrics = {'Loss': loss, **metrics}
        log_columns = ClassificationTask.wandb_log_columns(model.target_levels, 
                                              metrics, (valid_data is not None))
        wandb_run = ClassificationTask.wandb_init(train_data, valid_data, model, 
            optimizer, weight_schedule, sampler, loss, batch_size, epochs, 
            wandb_config, wandb_name)
        
        # Training
        t0 = time.time()
        print("Training...") if utils.VERBOSE > 0 else None
        for epoch in tqdm(range(epochs)):
            model.train()
            w = weight_schedule(epoch).to(utils.DEVICE)
            train_loss = torch.zeros(len(model.target_levels)).to(utils.DEVICE)
            for (x,y) in train_dataloader:
                # Make a prediction
                x, y = x.to(utils.DEVICE), y.to(utils.DEVICE)
                y_pred = model(x)
                # Learning step
                losses = torch.cat([loss[lvl](y_pred[i], y[:,lvl]).reshape(1)
                                  for i, lvl in enumerate(model.target_levels)])
                mean_loss = mean(losses, w)
                optimizer.zero_grad()
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                # Update metrics
                train_loss += x.size(0)*losses.detach()

            # Validation results
            scores = ClassificationTask.evaluate(model, train_data, metrics)
            scores = np.concatenate([[epoch+1], scores])
            if valid_data is not None:
                scores = np.concatenate([scores, 
                       ClassificationTask.evaluate(model, valid_data, metrics)])
            wandb_run.log({column: score 
                           for column, score in zip(log_columns, scores)})

        # Finishing the wandb_run, getting results dataframe
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        wandb_run.config.update({'num_params': params})
        wandb_run.finish(quiet=True)
        wandb_api = wandb.Api()
        run = wandb_api.run(f'{wandb_run.project}/{wandb_run._run_id}')
        history = run.history(pandas=True)
        
        if utils.VERBOSE > 0:
            print("Training finished, log saved to wandb (see above).")
            print("Final accuracy scores:\n---------------------\n")
            ClassificationTask.final_report(history,model.target_levels,'train')
            if valid_data is not None:
                ClassificationTask.final_report(history,model.target_levels,
                                                'valid')
        
        return model, history
    
    @staticmethod
    def test(model, data, loss=None, metrics=EVAL_METRICS):
        '''Produces results overview on test dataset (multiple test sets 
        supported when provided as list in `data`)'''

        # Initializing
        print('Evaluating...') if utils.VERBOSE > 0 else None
        data = [data] if type(data) != list else data
        if loss is None:
            loss = [torch.nn.CrossEntropyLoss(ignore_index=utils.UNKNOWN_INT) 
                    for i in range(6)]
        metrics = {'Loss': loss, **metrics}
        results = ClassificationTask.results_init(model.target_levels, metrics)
        coverage = {True: 'known', False: 'total'}
        
        for i in range(len(data)): # Looping over datasets and evaluating
            for ignore_uknowns in [True, False]:
                name = [f'Test set {i} ({coverage[ignore_uknowns]})']
                result = list(ClassificationTask.evaluate(model, data[i], 
                                                       metrics, ignore_uknowns))
                result = pd.DataFrame([name + result], columns=results.columns)
                results = pd.concat([results, result])

        results = results.set_index([''])
        if utils.VERBOSE > 0:
            results.to_csv(utils.OUTPUT_DIR + '/test_results.csv', 
                        float_format='{:.4}'.format)
            print("Test results saved.")
        return results

    @staticmethod
    def evaluate(model, data, metrics, ignore_unknowns=False):
        '''Evaluates performance of model for the specified `metrics`.
        
        Parameters
        ----------
        model: DeepITS
            The to-be-evaluated neural network
        data: UniteData
            Test data
        metrics: dict{str:function}
            Evaluation metrics to report during testing
        ignore_unknowns:
            Ignore the unrecognized labels in the dataset (default is False)'''

        model.eval()
        with torch.no_grad():
            y_pred, y = model._predict(data, return_labels=True)
            results = []
            for m in metrics:
                for i, lvl in enumerate(model.target_levels):
                    y_lvl, y_pred_i = y[:,lvl], y_pred[i]
                    if ignore_unknowns:
                        mask = y[:,lvl] != utils.UNKNOWN_INT
                        y_lvl, y_pred_i = y_lvl[mask], y_pred_i[mask]
                    if m == 'Loss':
                        results.append(metrics[m][lvl](y_pred_i, y_lvl).item())
                    else:
                        argmax_y_pred = torch.argmax(y_pred_i, dim=1)
                        results.append(metrics[m](y_lvl.cpu().numpy(), 
                                                  argmax_y_pred.cpu().numpy()))
        
        if len(model.target_levels) == 6:
            cons = ClassificationTask.consistency(y_pred, model.tax_encoder)
            return np.array(results + cons)
        else:
            return np.array(results)
        
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
    def results_init(target_levels, metrics):
        '''Initializes an empty results dataframe with correct rows/columns'''
        columns += [f'{metric}|{utils.LEVELS[lvl]}' for metric in metrics 
                                                       for lvl in target_levels]
        if len(target_levels) == 6:
            columns += ([f'Consistency|{pair}' 
                               for pair in ['P-C', 'C-O', 'O-F', 'F-G', 'G-S']])
        return pd.DataFrame(columns=columns)

    @staticmethod
    def wandb_log_columns(target_levels, metrics, use_valid):
        '''Returns a list of column names for the wandb log'''
        columns = ['Epoch']
        for dataset in ['train', 'valid'][:int(use_valid)+1]:
            columns +=  [f'{metric}|{dataset}|{utils.LEVELS[lvl]}' 
                           for metric in metrics for lvl in target_levels]
            if len(target_levels) == 6:
                columns += ([f'Consistency|{dataset}|{pair}' 
                               for pair in ['P-C', 'C-O', 'O-F', 'F-G', 'G-S']])
        return columns
        
    @staticmethod
    def wandb_init(train_data, valid_data, model, optimizer, weight_schedule, 
                   sampler, loss, batch_size, epochs, wandb_config, wandb_name):
        config = {
            **utils.get_config(train_data, prefix='trainset'),
            **utils.get_config(valid_data, prefix='validset'),
            **utils.get_config(model),
            **utils.get_config(optimizer, prefix='opt'),
            **utils.get_config(weight_schedule, 'weight_sched'),
            **utils.get_config(sampler, 'sampler'),
            **utils.get_config(loss[0], 'loss'),
            'batch_size': batch_size, 
            'epochs': epochs,
            **wandb_config
        }
        return wandb.init(project='MycoAI DeepITS classification',config=config, 
                          name=wandb_name, dir=utils.OUTPUT_DIR)

    @staticmethod
    def final_report(history, target_levels, train_or_valid):
        print(train_or_valid.capitalize() + ":")
        report = history[[f'Accuracy|{train_or_valid}|{utils.LEVELS[lvl]}' 
                          for lvl in target_levels]]
        columns = dict(zip(report.columns, 
                           [utils.LEVELS[lvl] for lvl in target_levels]))
        report = report.tail(1)
        report.rename(columns=columns, inplace=True)
        report.reset_index(drop=True, inplace=True)
        print(report)