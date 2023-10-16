''''For training and testing deep learning models on ITS classification task.'''

import time
import torch
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
                'Precision': partial(skmetric.precision_score, average='macro'),
                'Recall': partial(skmetric.recall_score, average='macro'),
                'F1': partial(skmetric.f1_score, average='macro'),
                'MCC': skmetric.matthews_corrcoef}

mean = lambda tensor, weights: (tensor @ weights) / weights.sum()


class ClassificationTask:
    '''Multi-task classification (for multiple taxonomic levels)'''

    @staticmethod
    def train(model, data, epochs, loss=torch.nn.CrossEntropyLoss(), 
              batch_size=64, optimizer=None, metrics=EVAL_METRICS, 
              valid_split=0.2, weight_schedule=None):
        '''Trains a neural network to classify ITS sequences
  
        Parameters
        ----------
        model: torch.nn.Module
            Neural network architecture
        data: mycoai.data.Dataset
            Preprocessed dataset (torch Dataset object) containing ITS sequences    
        epochs: int
            Number of training iterations
        batch_size: int
            Number of training examples per optimization step (default is 64)
        loss: list | function
            To-be-optimized loss function (or list of functions per level) 
            (default is CrossEntropyLoss)
        optimizer: torch.optim
            Optimization strategy (default is Adam)
        metrics: dict{str:function}
            Evaluation metrics to report during training, provided as dictionary
            with metric name as key and function as value (default is accuracy, 
            balanced acuracy, precision, recall, f1, and mcc).
        valid_split: float
            Data proportion be used for validation (default is 0.2)
        weight_schedule:
            Factors by which each level should be weighted in loss per epoch 
            (default is Constant([1,1,1,1,1,1]))'''
        
        # Initializing/setting parameter values
        if type(loss) != list:
            loss = [loss for i in range(len(model.target_levels))]
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, 
                                        weight_decay=0.0001)
        if weight_schedule is None:
            weight_schedule = ws.Constant([1]*len(model.target_levels))
        train_data, valid_data = tud.random_split(data, 
                                                  [1-valid_split, valid_split])
        train_dataloader = tud.DataLoader(train_data, batch_size=batch_size, 
                                          shuffle=True)
        metrics = {'Loss': loss} | metrics
        history = ClassificationTask.history_init(model.target_levels,
                                                           metrics, valid_split)

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
            scores = ((train_loss) / len(train_data)).detach().cpu().numpy()
            if valid_split > 0:
                scores = np.concatenate([scores, 
                       ClassificationTask.evaluate(model, valid_data, metrics)])
            scores = scores.reshape(1,-1)
            history = pd.concat([history, 
                                  pd.DataFrame(scores,columns=history.columns)],
                                  ignore_index=True)
            # TODO Currently only works when valid_data is available
            ClassificationTask.save_history(history,metrics,model.target_levels)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of parameters:", params)
        print("Training time (s): " + str(time.time()-t0))
        
        return model, history
    
    @staticmethod
    def test(model,data,loss=torch.nn.CrossEntropyLoss(), metrics=EVAL_METRICS):
        '''Produces results overview on test dataset (multiple test sets 
        supported when provided as list in `data`)'''

        # Initializing
        print('Evaluating...') if utils.VERBOSE > 0 else None
        data = [data] if type(data) != list else data
        if type(loss) != list:
            loss = [loss for i in range(len(model.target_levels))]
        metrics = {'Loss': loss} | metrics
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
        model: ITSClassifier
            The to-be-evaluated neural network
        data: UniteData
            Test data
        metrics: dict{str:function}
            Evaluation metrics to report during testing
        ignore_unknowns:
            Ignore the unrecognized labels in the dataset (default is False)'''

        model.eval()
        with torch.no_grad():
            y_pred, y = model.predict(data, return_labels=True)
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
        columns = ['']
        columns += [f'{metric}|{utils.LEVELS[lvl]}' for metric in metrics 
                                                       for lvl in target_levels]
        if len(target_levels) == 6:
            columns += ([f'Consistency|{pair}' 
                               for pair in ['P-C', 'C-O', 'O-F', 'F-G', 'G-S']])
        return pd.DataFrame(columns=columns)

    @staticmethod
    def history_init(target_levels, metrics, valid_split):
        '''Initializes an empty history dataframe with correct columns'''
        columns = [f'Loss|train|{utils.LEVELS[lvl]}' for lvl in target_levels]
        if valid_split > 0:
            columns +=  [f'{metric}|valid|{utils.LEVELS[lvl]}' 
                           for metric in metrics for lvl in target_levels]
        if valid_split > 0 and len(target_levels) == 6:
            columns += ([f'Consistency|valid|{pair}' 
                               for pair in ['P-C', 'C-O', 'O-F', 'F-G', 'G-S']])
        return pd.DataFrame(columns=columns)

    @staticmethod
    def save_history(history, metrics, target_levels):
        '''Exports history file (verbose > 0), generates plots (verbose > 1)'''
        if utils.VERBOSE > 0:
            history.to_csv(utils.OUTPUT_DIR + '/train_scores.csv', 
                           float_format='{:.3}'.format)
            print("Training loss history saved.")
        if utils.VERBOSE > 1:
            plotter.classification_loss(history, target_levels)
            for metric in metrics:
                if metric != 'Loss':
                    plotter.classification_metric(history, metric,target_levels)
            print("Training metric plots saved.")
        