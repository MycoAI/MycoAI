'''Example script'''

import torch
from mycoai import data, utils, plotter, encoders
from mycoai.models import DeepITS
from mycoai.models.architectures import ResNet, SimpleCNN, BERT
from mycoai.training import ClassificationTask, MLMTask


# Some settings
utils.set_output_dir('results') # Create results directory, save output there
utils.set_device('cuda:1')
utils.VERBOSE = 2

# Data import
utils.set_output_dir('results')
train_data = data.DataPrep('/data/s2592800/test2.fasta') # Parse file
# plotter.counts_sunburstplot(train_data, id='test1')
# train_data = train_data.class_filter('species', min_samples=5)
# train_data = train_data.sequence_length_filter()
# train_data = train_data.sequence_quality_filter()
# plotter.counts_sunburstplot(train_data.data, id='filtered')
train_data, valid_data = train_data.encode_dataset(dna_encoder='bpe', valid_split=0.2)
# train_data, valid_data = train_data.encode_dataset(dna_encoder='kmer-spectral', valid_split=0.2)


# MLMTask.train(None, train_data)


# Use encoding scheme from train_data on the test set
# test_data = data.DataPrep('/data/s2592800/test1.fasta')
# test_data = test_data.encode_dataset(tax_encoder=train_data.tax_encoder)

# Model definition
# arch = SimpleCNN(kernel=5,conv_layers=[5,10],in_channels=1, pool_size=2)
# arch = ResNet([2,2,2,2], in_channels=4) # = ResNet18
arch = BERT(train_data.dna_encoder.vocab_size, d_model=64, d_ff=256)
model = DeepITS(arch, train_data.dna_encoder, train_data.tax_encoder,
                                     fcn_layers=[128,20,64], output='multi')

# Train/test with weighted loss for class imbalance
sampler = train_data.weighted_sampler('species')
loss_functions = train_data.weighted_loss(torch.nn.CrossEntropyLoss, sampler=sampler)
model, history = ClassificationTask.train(model, train_data, valid_data, 100, sampler=sampler, loss=loss_functions)
torch.save(model, 'models/test2.pt')
plotter.classification_loss(history, model.target_levels)
# result = ClassificationTask.test(model, test_data)