'''Inference and evaluation of deep learning methods'''

import torch
import time
from mycoai import utils
from mycoai.data import Data
from mycoai.evaluate import Evaluator

utils.set_output_dir('/data1/s2592800/')
utils.WANDB_PROJECT = 'Final Comparison'
utils.PRED_BATCH_SIZE = 64

model_folder = '/data1/s2592800/models/'
data_folder = '/data1/s2592800/'
results_folder = '/data1/s2592800/comparison/deep/'

models = ['MycoAI-multi-BERT-medium-HLS.pt', # Multi-head BERT, LS variations
          'MycoAI-multi-BERT-medium-NoLS.pt',
          'MycoAI-multi-BERT-medium-SLS.pt',
          'MycoAI-single-BERT-medium-HLS.pt', # Single-head BERT, LS variations
          'MycoAI-single-BERT-medium-NoLS.pt',
          'MycoAI-single-BERT-medium-SLS.pt',
          'MycoAI-multi-CNN-Vu-HLS.pt', # Multi-Head CNN, LS variations
          'MycoAI-multi-CNN-Vu-NoLS.pt',
          'MycoAI-multi-CNN-Vu-SLS.pt',
          'MycoAI-single-CNN-Vu-HLS.pt', # Single-head CNN, LS variations
          'MycoAI-single-CNN-Vu-NoLS.pt',
          'MycoAI-single-CNN-Vu-SLS.pt',
          'MycoAI-multi-CNN-Vu-NoBN-HLS.pt', # Multi-head CNN, arch variations
          'MycoAI-multi-CNN-ResNet9-HLS.pt',
          'MycoAI-multi-CNN-ResNet18-HLS.pt',
          'MycoAI-multi-BERT-small-HLS.pt', # Multi-head BERT, arch variations
          'MycoAI-multi-BERT-large-HLS.pt',
          'MycoAI-multi-BERT-medium-4mer-HLS.pt', # Multi-head BERT, encodings
          'MycoAI-multi-BERT-medium-5mer-HLS.pt',
          'MycoAI-multi-BERT-medium-6mer-HLS.pt'
          ]

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

models = ['MycoAI-multi-BERT-medium-HLS.pt',
          'MycoAI-multi-BERT-medium-NoLS.pt',
          'MycoAI-multi-BERT-medium-SLS.pt',
          'MycoAI-multi-CNN-Vu-HLS.pt',
          'MycoAI-multi-CNN-Vu-NoLS.pt',
          'MycoAI-multi-CNN-Vu-SLS.pt']

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