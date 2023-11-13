'''Evaluation and analysis of ITS classification algorithms.'''

import wandb
import sklearn
import pandas as pd
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
        reference: mycoai.Data
            Reference dataset with true labels.
        classifier: 
            If provided and equipped with a .get_config method, will add its 
            configuration information to the wandb run. Default is None.
        wandb_config: dict
            Extra information to be added to weights and biases config data. 
        wandb_name: str
            Name of the run to be displayed on weights and biases.'''
        
        self.pred = classification
        self.true = reference
        self.wandb_run = self.wandb_init(reference, classifier, wandb_config, 
                                         wandb_name)

    def test(self, metrics=utils.EVAL_METRICS, target_levels=utils.LEVELS):
        '''Calculates classification performance in terms of specified metrics.
        
        Parameters
        ----------
        metrics: dict{str: function}
            To-be-evaluated metrics. Provided as dictionary with metrics names
            as keys (str) and a callable fucntion as value.
        target_levels: list[str]
            Taxonomic levels at which the predictor should be evaluated. Default
            is [phylum, class, order, family, genus, species]. '''
        
        print('Evaluating...') if utils.VERBOSE > 0 else None
        results = {}
        for m in metrics:
            for lvl in target_levels:
                name = f'{m}|test|{lvl}'
                value = metrics[m](self.true.data[lvl], self.pred[lvl])
                results[name] = value

        self.wandb_run.log(results, step=0)
        classifier = self.wandb_run.config.get('type', 0)
        info = {'Dataset': self.wandb_run.config['testset_name']}
        info = {'Classifier': classifier, **info} if classifier != 0 else info
        results = {
            **info,
            **{key:[results[key]] for key in results}
        }

        return pd.DataFrame(results)

    def confusion_matrix(self, level):
        '''Creates confusion matrix at specified level'''
        encoder = sklearn.preprocessing.LabelEncoder()
        encoder.fit(pd.concat([self.pred[level], self.true.data[level]]))
        pred = encoder.transform(self.pred[level])
        true = encoder.transform(self.true.data[level])
        conf_mat = wandb.plot.confusion_matrix(preds=pred, y_true=true, 
                                       probs=None, class_names=encoder.classes_)
        wandb.log({f"Confusion|test|{level}" : conf_mat}, step=0)
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
        self.wandb_run.finish(quiet=True)