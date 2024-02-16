'''Inference of DNABarcoder and RDPClassifier'''

import pandas as pd
from mycoai import utils
from mycoai.data import Data
from mycoai.evaluate import Evaluator

utils.WANDB_PROJECT = 'Comparison'

data_folder = '/data/luuk/'
results_folder = '/data/luuk/results/'

testsets = ['test1.fasta',
            'test2.fasta',
            'trainset_valid.fasta']

# Evaluating DNABarcoder
for testset_name in testsets:

    name = testset_name[:-6] + " DNABarcoder"

    # Convert to csv and format properly
    classification = pd.read_csv(
        f'{results_folder}dnabarcoder/{testset_name[:-6]}' + 
        f'_prepped.trainset_identified_BLAST.classification', delimiter='\t'
    )
    classification = classification[utils.LEVELS]
    classification['species'] = classification['species'].str.replace(' ', '_')

    # Evaluate
    reference = Data(data_folder + testset_name, allow_duplicates=True)
    evaluator = Evaluator(classification, reference, 
                          wandb_name=f'Results {name}')
    results = evaluator.test()
    results.to_csv(results_folder + 'results/' + name + '.csv')
    for level in utils.LEVELS:
        report = evaluator.detailed_report(level)
        report.to_csv(f'{results_folder}results/{name} {level} report.csv')
    evaluator.wandb_finish()

# Evaluating RDP classifier predictions
for testset_name in testsets:

    name = testset_name[:-6] + " RDP"

    # Convert to csv and format properly
    classification = pd.read_csv(f'{results_folder}rdp/{testset_name[:-6]}.txt',
                                 delimiter='\t', header=None)
    classification = classification[[8,11,14,17,20,23]]
    classification.columns = utils.LEVELS
    classification['species'] = classification['species'].str.split('|').str[0]

    # Evaluate
    reference = Data(data_folder + testset_name, allow_duplicates=True)
    evaluator = Evaluator(classification, reference, 
                          wandb_name=f'Results {name}')
    results = evaluator.test()
    results.to_csv(results_folder + 'results/' + name + '.csv')
    for level in utils.LEVELS:
        report = evaluator.detailed_report(level)
        report.to_csv(f'{results_folder}results/{name} {level} report.csv')
    evaluator.wandb_finish()