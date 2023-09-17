'''Example script'''

import torch
from mycoai import data, utils
from mycoai.models import ITSClassifier
from mycoai.models.architectures import ResNet
from mycoai.training import ClassificationTask

# Some settings
utils.set_output_dir('results') # Create results directory, save output there
utils.set_device('cuda:0') # Use cpu
utils.VERBOSE = 2

# Data import
train_data = data.DataPrep('/data/s2592800/test2.fasta') # Parse file
train_data = train_data.encode_dataset()
# train_data = train_data.encode_dataset(export_path='/data/s2592800/test.pt')
                                                    #  Export to save time later
# Use encoding scheme from train_data on the test set
test_data = data.DataPrep('/data/s2592800/test1.fasta')
test_data = test_data.encode_dataset(tax_encoder=train_data.tax_encoder)

# Model definition
arch = ResNet([2,2,2,2]) # = ResNet18
model = ITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder, target_levels=['class'],
                                     fcn_layers=[128,50,64], output='single')

# Train/test with weighted loss for class imbalance
loss_functions = train_data.weighted_loss(torch.nn.CrossEntropyLoss)
model, history = ClassificationTask.train(model,train_data, 100, loss_functions)
result = ClassificationTask.test(model, test_data, loss_functions)