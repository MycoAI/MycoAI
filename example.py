'''Example script'''

import torch
from mycoai import data, utils, plotter
from mycoai.models import ITSClassifier
from mycoai.models.architectures import ResNet
from mycoai.training import ClassificationTask

# Some settings
utils.set_output_dir('results') # Create results directory, save output there
# utils.set_device('cuda') # To specify GPU use
# utils.VERBOSE = 1 # To turn on/off prints/plots

# Data import & preprocessing
train_data = data.DataPrep('/data/s2592800/test1.fasta')
train_data = train_data.class_filter('species', min_samples=5)
train_data = train_data.sequence_length_filter()
train_data = train_data.sequence_quality_filter()
train_data, valid_data = train_data.encode_dataset('4d', valid_split=0.2)

# Use encoding scheme from train_data on the test set
test_data = data.DataPrep('/data/s2592800/test2.fasta')
test_data = test_data.encode_dataset(dna_encoder=train_data.dna_encoder,
                                     tax_encoder=train_data.tax_encoder)

# Model definition
arch = ResNet([2,2,2,2]) # = ResNet18
# This model will have a single output head and make genus-level predictions
model = ITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder,  
               target_levels=['genus'], fcn_layers=[128,20,64], output='single')

# Train/test (optionally with weighted loss/sampling)
# sampler = train_data.weighted_sampler('genus')
# loss_function = train_data.weighted_loss(torch.nn.CrossEntropyLoss,
#                                          sampler=sampler)
model, history = ClassificationTask.train(model, train_data, valid_data, 100)
plotter.classification_loss(history, model.target_levels)
result = ClassificationTask.test(model, test_data)