'''Example script that demonstrates how to use the MycoAI package'''

import torch
from mycoai import utils
from mycoai.data import Data
from mycoai.evaluate import Evaluator
from mycoai.modules import SeqClassNetwork, BERT
from mycoai.train import SeqClassTrainer, CrossEntropyLoss

# Some settings
# utils.set_device('cpu') # Uncomment to force CPU use 

# Training data import & preprocessing
train_data = Data('data/trainset_valid.fasta')
train_data = train_data.sequence_length_filter(tolerance=4)
train_data = train_data.sequence_quality_filter(tolerance=0.05)
train_data, valid_data = train_data.train_valid_split(0.2)

# Encoding the datasets, use encoders from training data for validation data
train_data = train_data.encode_dataset('bpe') # Byte Pair Encoding
valid_data = valid_data.encode_dataset(train_data.dna_encoder, 
                                       train_data.tax_encoder)

# Model = encoders + base architecture + output layer
arch = BERT(train_data.dna_encoder.vocab_size) # Base architecture
model = SeqClassNetwork(arch, train_data.dna_encoder, train_data.tax_encoder,
                        output='multi') # Multi-head output

# Train the network (with weighted loss)
loss = train_data.weighted_loss(CrossEntropyLoss, strength=0.5)
model, history = SeqClassTrainer.train(model, train_data, valid_data, loss=loss)
torch.save(model, 'model.pt') # Export model for later use

# Make a prediction on the test set
test_data = Data('data/test1.fasta', allow_duplicates=True)
classification = model.classify(test_data)
evaluator = Evaluator(classification, test_data, model)
evaluator.test()