'''Example script'''

import torch
from mycoai import data, utils
from mycoai.evaluate import Evaluator
from mycoai.deep.models import DeepITSClassifier
from mycoai.deep.models import ResNet
from mycoai.deep.train import DeepITSTrainer

# Some settings
utils.set_output_dir('results') # Create results directory, save output there
# utils.set_device('cpu') # To specify CPU/GPU use

# Data import & preprocessing
train_data = data.Data('test1.fasta')
train_data = train_data.class_filter('species', min_samples=5)
train_data = train_data.sequence_length_filter()
train_data = train_data.sequence_quality_filter()
train_data, valid_data = train_data.encode_dataset('4d', valid_split=0.2)

# Use encoding scheme from train_data on the test set
test_data = data.Data('test2.fasta', allow_duplicates=True)
test_data = test_data.encode_dataset(dna_encoder=train_data.dna_encoder,
                                     tax_encoder=train_data.tax_encoder)

# Model definition
arch = ResNet([2,2,2,2]) # = ResNet18
# This model will have a single output head and make genus-level predictions
model = DeepITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder,  
               target_levels=['genus'], fcn_layers=[128,20,64])

# Train (optionally with weighted loss/sampling)
# sampler = train_data.weighted_sampler('genus')
# loss_function = train_data.weighted_loss(torch.nn.CrossEntropyLoss,
#                                          sampler=sampler)
model, history = DeepITSTrainer.train(model, train_data, valid_data, 100)

# Make a prediction on the test set
classification = model.predict(test_data)
evaluator = Evaluator(classification, test_data, model)
evaluator.test()