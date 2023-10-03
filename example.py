'''Example script'''

import torch
from mycoai import data, utils, plotter
from mycoai.models import ITSClassifier
from mycoai.models.architectures import ResNet, SimpleCNN
from mycoai.training import ClassificationTask

# Some settings
utils.set_output_dir('results') # Create results directory, save output there
# utils.set_device('cuda') # To specify GPU use 
# utils.VERBOSE = 1 # To turn on/off prints/plots

# Data import & preprocessing
train_data = data.DataPrep('trainset.fasta')
train_data = train_data.class_filter('species', min_samples=5)
train_data = train_data.sequence_length_filter()
train_data = train_data.sequence_quality_filter()
train_data, valid_data = train_data.encode_dataset(valid_split=0.2)  
                                                 # , dna_encoder='spectral') # = Duong's model

# Use encoding scheme from train_data on the test set
test_data = data.DataPrep('testset.fasta')
test_data = test_data.encode_dataset(tax_encoder=train_data.tax_encoder,
                                     dna_encoder=train_data.dna_encoder)

# Model definition
# arch = SimpleCNN(kernel=5,conv_layers=[5,10],in_channels=1,pool_size=2) # = Duong's model
arch = ResNet([2,2,2,2]) # = ResNet18
# This model will have a single output head and make genus-level predictions
model = ITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder,  
               target_levels=['genus'], fcn_layers=[128,20,64], output='single')

# Train/test (optionally with weighted loss/sampling) 
# sampler = train_data.weighted_sampler('genus')
# loss_function = train_data.weighted_loss(torch.nn.CrossEntropyLoss, 
#                                          sampler=sampler)
model, history = ClassificationTask.train(model, train_data, valid_data, 100)
# plotter.classification_loss(history, model.target_levels)
result = ClassificationTask.test(model, test_data)