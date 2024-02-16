'''Inference and evaluation of deep learning methods'''

import torch
import time
from mycoai import utils
from mycoai.data import Data
from mycoai.evaluate import Evaluator

utils.WANDB_PROJECT = 'Comparison'
utils.PRED_BATCH_SIZE = 64

model_folder = '/data1/s2592800/models/'
data_folder = '/data1/s2592800/'
results_folder = '/data1/s2592800/comparison/deep/'

models = ['MycoAI-multi LS 0.02 0.02 0.02 0.02 0.02 0.0.pt',
          'MycoAI-multi LS 0.0 0.0 0.0 0.0 0.0 0.1.pt',
          'MycoAI-single LS 0.02 0.02 0.02 0.02 0.02 0.0.pt',
          'MycoAI-single LS 0.0 0.0 0.0 0.0 0.0 0.1.pt',]

testsets = ['test1.fasta',
            'test2.fasta',
            'test3.fasta',
            'trainset_valid.fasta']

train_src = Data(data_folder + 'trainset.fasta')

for model_name in models:

    for testset_name in testsets:

        # Create name
        name = testset_name[:-6] + " " + model_name[:-3]

        # Make and time prediction (data loading/saving included)
        t0 = time.time()
        model = torch.load(model_folder + model_name, map_location=utils.DEVICE)
        classification = model.classify(data_folder + testset_name)
        classification.to_csv(results_folder + 'predictions/' + name + '.csv')
        t1 = time.time()
        print(f'Runtime {name}: {t1-t0} s')

        # No need to analyze performance on test set 3
        if testset_name == 'test3.fasta':
            del model, classification
            torch.cuda.empty_cache()
            continue

        # Analyze results
        reference = Data(data_folder + testset_name, allow_duplicates=True)
        latent_space = None
        if testset_name != 'test3.fasta':
            latent_space = model.latent_space(reference)
        evaluator = Evaluator(classification, reference, model,
                              wandb_name=f'Results {name}')
        results = evaluator.test()
        results.to_csv(results_folder + 'results/' + name + '.csv')
        # report = evaluator.detailed_report('species', train_src, latent_space)
        # report.to_csv(f'{results_folder}results/{name} species report.csv')
        evaluator.wandb_finish()

        del model, classification, latent_space, reference, evaluator
        del results # , report
        torch.cuda.empty_cache()

models = ['MycoAI-multi LS 0.02 0.02 0.02 0.02 0.02 0.0.pt',
          'MycoAI-multi LS 0.0 0.0 0.0 0.0 0.0 0.1.pt']

# Compare hierarchical to standard label smoothing
for model_name in models:

    if model_name.startswith('MycoAI-multi'):

        name = model_name[:-3] + " parents inferred"
        model = torch.load(model_folder + model_name, map_location=utils.DEVICE)
        model.multi_to_infer_sum() # Change output type to InferSum
        classification = model.classify(data_folder + 'trainset_valid.fasta')
        classification.to_csv(results_folder + 'predictions/' + name + '.csv')
        reference = Data(data_folder + 'trainset_valid.fasta')
        evaluator = Evaluator(classification, reference, model, 
                              wandb_name=f'Results {name}')
        results = evaluator.test()
        results.to_csv(results_folder + 'results/' + name + '.csv')
        evaluator.wandb_finish()

        del model, classification, reference, evaluator, results
        torch.cuda.empty_cache()

