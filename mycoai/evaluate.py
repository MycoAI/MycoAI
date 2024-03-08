'''Evaluation and analysis of sequence classification algorithms.'''

import wandb
import sklearn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mycoai import utils

class Evaluator: 
    '''Object with multiple evaluation/analysis methods, initialized for a
    specific classifier-dataset combination.'''

    def __init__(self, classification, reference, classifier=None, 
                 wandb_config={}, wandb_name=None):
        '''Initializes Evaluator instance for specified dataset (and model).
        
        Parameters
        ----------
        classification: pd.DataFrame
            The to-be-evaluated classification (prediction)
        reference: pd.DataFrame | mycoai.Data
            Reference dataset with true labels.
        classifier: 
            If provided and equipped with a .get_config method, will add its 
            configuration information to the wandb run. Default is None.
        wandb_config: dict
            Extra information to be added to weights and biases config data. 
        wandb_name: str
            Name of the run to be displayed on weights and biases.'''
        
        self.pred = classification
        if type(reference) == pd.DataFrame:
            self.true = reference
        else: # mycoai.Data
            self.true = reference.data
        self.wandb_run = self.wandb_init(reference, classifier, wandb_config, 
                                         wandb_name)

    def test(self, metrics=utils.EVAL_METRICS, levels=utils.LEVELS):
        '''Calculates classification performance in terms of specified metrics.
        
        Parameters
        ----------
        metrics: dict{str: function}
            To-be-evaluated metrics. Provided as dictionary with metrics names
            as keys (str) and a callable fucntion as value.
        levels: list[str]
            Taxonomic levels at which the predictor should be evaluated. Default
            is [phylum, class, order, family, genus, species]. '''
        
        print('Evaluating...') if utils.VERBOSE > 0 else None
        results = {}
        for m in metrics:
            for lvl in levels:
                # Only consider samples that are known in the reference
                known = self.true[lvl] != utils.UNKNOWN_STR
                name = f'{m}|test|{lvl}'
                value = metrics[m](self.true[known][lvl], self.pred[known][lvl])
                results[name] = value

        self.wandb_run.log(results, step=0)
        self.metrics_bars(results, metrics, levels)
        classifier = self.wandb_run.config.get('type', 0)
        info = {'Dataset': self.wandb_run.config['testset_name']}
        info = {'Classifier': classifier, **info} if classifier != 0 else info
        results = pd.DataFrame({
            **info,
            **{key:[results[key]] for key in results}
        })

        if utils.VERBOSE > 0:
            self.test_report(results, metrics)
        return results
    
    def test_report(self, results, metrics):
        '''Prints results from test method.'''
        
        table = []
        for m in metrics:
            table.append([m] + list(
                results[[f'{m}|test|{lvl}' for lvl in utils.LEVELS]].values[0]
            ))
        table = pd.DataFrame(table, columns=[['Metric'] + utils.LEVELS])
        print(table)

    def metrics_bars(self, results, metrics, levels):
        '''Plots evaluation results in custom chart on WandB'''
        for m in metrics:
            data = [results[f'{m}|test|{lvl}'] for lvl in levels]
            table = wandb.Table(
                data=pd.DataFrame({'Level': levels, m: data}))
            self.wandb_run.log({f'Overview/{m} table': 
                wandb.plot.bar(table, 'Level', m, f'{m} per level')}, step=0)
            
    def detailed_report(self, level='species', train_data=None, 
                        latent_repr=None):
        '''Provides detailed information (on WandB) for one specified level.'''

        data = self.label_confusion_info(level)
        data = self.label_frequency_info(data, level, train_data)
        data = data.fillna(0.0) # Replace NaN's

        # Call the visualization methods
        self.metric_per_class(level, data, 'f1')
        if train_data is not None:
            self.metric_per_train_freq(level, data, 'f1')
        self.mispredictions_per_class(level, data)
        self.frequency_comparison(level, data)
        if latent_repr is not None:
            self.latent_space_visualization(level, data, latent_repr)
        
        return data

    def label_confusion_info(self, level):
        '''Returns number of True/False positives/negatives, precision, recall,
        and F1 per label at specified level.'''

        # Initializing
        data = []
        conf_matrix = sklearn.metrics.confusion_matrix(y_true=self.true[level], 
                                                       y_pred=self.pred[level])
        labels = np.unique(np.concatenate((self.true[level].values, 
                                           self.pred[level].values)))
        
        for i, label in enumerate(labels):

            # Calculate true/false positives/negatives and associated metrics
            TP = conf_matrix[i,i]
            FP = conf_matrix[:,i].sum() - TP
            TN = (conf_matrix.sum() - conf_matrix[:,i].sum() - 
                  conf_matrix[i,:].sum() + TP)
            FN = conf_matrix[i,:].sum() - TP
            precision = TP / (TP + FP) 
            recall = TP / (TP + FN)
            f1 = 2*((precision * recall) / (precision + recall))

            # Calculate which label the model is most often confused with
            row = conf_matrix[i].copy()
            row[i] = 0
            confused_with = np.argmax(row)
            if row[confused_with] != 0:
                confused_with = labels[confused_with]
            else:
                confused_with = '-' 

            data.append([label, TP, FP, TN, FN, confused_with, precision, 
                         recall, f1])
            
        columns = [level, 'TP', 'FP', 'TN', 'FN', 'most confused with', 
                   'precision', 'recall', 'f1']
        return pd.DataFrame(data, columns=columns)

    def label_frequency_info(self, label_data, level, train_data=None):
        '''Adds frequency (occurrence in test/trainset) to label_data.'''

        # Count occurrence in test/trainset
        count_pred = self.pred[[level]].groupby(level, as_index=False).size()
        data = pd.merge(label_data, count_pred, 'left', level)
        count_true = self.true[[level]].groupby(level, as_index=False).size()
        data = pd.merge(data, count_true, 'left', level)
        data = data.rename(columns={'size_x':'Count (pred.)', 
                                    'size_y':'Count (true)'})
        if train_data is not None:
            count_train = train_data.data.groupby(level, as_index=False).size()
            data = pd.merge(data, count_train, 'left', level)
            data = data.rename(columns={'size':'Count (train)'})
        
        return data
    
    def metric_per_class(self, level, data, metric):
        '''Plots the metric data per class at the specified level.'''

        fig = px.scatter(data, x=range(len(data)), y=metric, 
                         hover_data=data.columns, 
                         title=f'{metric.capitalize()} per {level}', 
                         labels={'x':level})

        # Push to wandb
        fig.write_html('temp.html', auto_play=False)
        table = wandb.Table(columns=['plotly_figure'])
        table.add_data(wandb.Html('temp.html'))
        self.wandb_run.log({f'Report/{metric}_per_{level}': table}, step=0)
        utils.remove_file('temp.html')

    def metric_per_train_freq(self, level, data, metric):
        '''Plots the metric data per occurrence frequency in the train set.'''

        fig = px.scatter(data, x='Count (train)', y=metric, 
                         hover_data=data.columns, 
                         title=f'{metric.capitalize()} per #training samples', 
                         labels={'x':level})

        # Push to wandb
        fig.write_html('temp.html', auto_play=False)
        table = wandb.Table(columns=['plotly_figure'])
        table.add_data(wandb.Html('temp.html'))
        self.wandb_run.log({f'Report/{metric}_per_trainfreq_{level}': table}, 
                           step=0)
        utils.remove_file('temp.html')

    def mispredictions_per_class(self, level, data):
        '''Plots the number of mispredictions per class at specified level.'''

        data['Incorrect'] = data['FP'] + data['FN']
        fig = px.scatter(data, x=range(len(data)), y='Incorrect', 
                         hover_data=data.columns, 
                         title=f'Mispredictions per {level}',
                         labels={'x':level})

        # Push to wandb
        fig.write_html('temp.html', auto_play=False)
        table = wandb.Table(columns=['plotly_figure'])
        table.add_data(wandb.Html('temp.html'))
        self.wandb_run.log({f'Report/mispredictions_per_{level}':table},
                           step=0)
        utils.remove_file('temp.html')

    def frequency_comparison(self, level, data):
        '''Plots the frequency of classes at the specified level in the
        test set vs the frequency in the prediction.'''

        # Calculate frequencies
        fig = px.scatter(data, f'Count (true)', 'Count (pred.)', 
                         hover_data=data.columns,
                         title=f'Occurrence frequency ({level})')

        fig.write_html('temp.html', auto_play=False)
        table = wandb.Table(columns=['plotly_figure'])
        table.add_data(wandb.Html('temp.html'))
        self.wandb_run.log({f'Report/frequency_pred_vs_true_{level}': 
                            table}, step=0)
        utils.remove_file('temp.html')

        return fig
    
    def latent_space_visualization(self, level, data, latent_repr, 
                                   color_by='phylum'):
        '''Visualizes latent space'''

        # Get data per-sample
        data = pd.merge(self.true[utils.LEVELS], data, how='left', on=level)
        data[f'predicted {level}'] = self.pred[level]

        # Reduce dimensions
        if latent_repr.shape[1] > 50: # Apply PCA first if dim>50 (recuce noise)
            latent_repr = PCA(50).fit_transform(latent_repr)
        latent_repr = TSNE().fit_transform(latent_repr) # Apply t-SNE, dim -> 2
        data['Dim 1'] = latent_repr[:,0].tolist()
        data['Dim 2'] = latent_repr[:,1].tolist()

        fig = px.scatter(data, x='Dim 1', y='Dim 2',hover_data=data.columns,
                         color=color_by)

        fig.write_html('temp.html', auto_play=False)
        table = wandb.Table(columns=['plotly_figure'])
        table.add_data(wandb.Html('temp.html'))
        self.wandb_run.log({f'Report/latent_space_{level}': 
                            table}, step=0)
        utils.remove_file('temp.html')

        return latent_repr

    def confusion_matrix(self, level):
        '''Creates confusion matrix at specified level'''
        encoder = sklearn.preprocessing.LabelEncoder()
        encoder.fit(pd.concat([self.pred[level], self.true[level]]))
        pred = encoder.transform(self.pred[level])
        true = encoder.transform(self.true[level])
        conf_mat = wandb.plot.confusion_matrix(preds=pred, y_true=true, 
                                       probs=None, class_names=encoder.classes_)
        wandb.log({f"Confusion/Confusion|test|{level}" : conf_mat}, step=0)
        return conf_mat.table.get_dataframe()  
    
    def wandb_init(self, data, classifier, wandb_config, wandb_name):
        '''Initializes wandb_run, writes config'''
        utils.wandb_cleanup()
        model_config={} if classifier is None else utils.get_config(classifier)
        config = {
            'task': 'evaluation',
            **utils.get_config(data, prefix='testset'),
            **model_config,
            **wandb_config,
            **utils.get_config()
        }
        return wandb.init(project=utils.WANDB_PROJECT, config=config, 
                          name=wandb_name, dir=utils.OUTPUT_DIR)
    
    def wandb_finish(self):
        '''Call this when including multiple wandb runs in one script'''
        self.wandb_run.finish(quiet=True)