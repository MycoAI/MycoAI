'''Example script'''

import torch
from mycoai import data, utils
from mycoai.models import ITSClassifier
from mycoai.models.architectures import ResNet
from mycoai.training import ClassificationTask

# Some settings
utils.set_output_dir('results') # Create results directory, save output there

# Data import
train_data = data.DataPrep('trainset.fasta') # Parse file
train_data = train_data.encode_dataset()
# Use encoding scheme from train_data on the test set
test_data = data.DataPrep('testset.fasta')
test_data = test_data.encode_dataset(tax_encoder=train_data.tax_encoder)

# Model definition
arch = ResNet([2,2,2,2]) # = ResNet18
model = ITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder,
                                     fcn_layers=[128,50,64], output='chained')

# Train/test with weighted loss for class imbalance
loss_functions = train_data.weighted_loss(torch.nn.CrossEntropyLoss)
model, history = ClassificationTask.train(model,train_data, 100, loss_functions)
result = ClassificationTask.test(model, test_data, loss_functions)