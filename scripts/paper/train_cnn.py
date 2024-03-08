import torch
from mycoai import utils
from mycoai.train import SeqClassTrainer
from mycoai.train.loss import CrossEntropyLoss
from mycoai.modules import ResNet, SimpleCNN, SeqClassNetwork
from mycoai.data import Data, TensorData
from mycoai.evaluate import Evaluator

valid_src = Data('/data1/s2592800/trainset_valid.fasta', allow_duplicates=True)
train_src = Data('/data1/s2592800/trainset.fasta', allow_duplicates=True) 
train = train_src.encode_dataset('kmer-spectral') 
valid = valid_src.encode_dataset(train.dna_encoder, train.tax_encoder)

output = 'multi'
lr = 0.0001
wd = 0

smoothing = [[0.020, 0.020, 0.020, 0.020, 0.020, 0.000], # Hierarchical
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000], # None
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.100], # Random
             [0.040, 0.040, 0.040, 0.040, 0.040, 0.000]] # More

for label_smoothing in smoothing:

    name = f'MycoAI-CNN LS {" ".join([str(s) for s in label_smoothing])}' 

    arch = SimpleCNN(kernel=5,conv_layers=[5,10],in_channels=1, pool_size=2)
    # arch = ResNet([1,1,1,1], in_channels=1)
    model = SeqClassNetwork(arch, train.dna_encoder, train.tax_encoder, 
                              fcn_layers=[256], output=output)

    loss = train.weighted_loss(CrossEntropyLoss, strength=0.5)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    try:
        model, _ = SeqClassTrainer.train(model, train, valid, 50, loss, 
            levels=[1,1,1,1,1,1], optimizer=optim, 
            label_smoothing=label_smoothing, wandb_name=name)
        
        classification = model.classify(valid)
        latent_repr = model.latent_space(valid)
        evaluator = Evaluator(classification, valid_src, model, 
                              wandb_name=name + ' (test)')
        evaluator.test()
        for level in utils.LEVELS:
            evaluator.detailed_report(level, train_src, latent_repr)
        evaluator.wandb_finish()
        torch.save(model, f'/data1/s2592800/{name}.pt')
    except Exception as exception:
        print("Failed:", exception)