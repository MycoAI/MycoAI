import torch
from mycoai import utils
from mycoai.deep.train import DeepITSTrainer
from mycoai.deep.train.loss import CrossEntropyLoss
from mycoai.deep.models import BERT, DeepITSClassifier
from mycoai.data import Data, TensorData
from mycoai.evaluate import Evaluator

valid_src = Data('/data1/s2592800/trainset_valid.fasta', allow_duplicates=True)
train_src = Data('/data1/s2592800/trainset.fasta', allow_duplicates=True)
train = TensorData(filepath='/data1/s2592800/trainset_bpe.pt')
valid = TensorData(filepath='/data1/s2592800/trainset_bpe_valid.pt')

output = 'infer_sum'
lr = 0.0001
wd = 0

smoothing = [[0.020, 0.020, 0.020, 0.020, 0.020, 0.000], # Hierarchical
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000], # None
             [0.000, 0.000, 0.000, 0.000, 0.000, 0.100], # Random
             [0.040, 0.040, 0.040, 0.040, 0.040, 0.000]] # More

for label_smoothing in smoothing:

    name = f'MycoAI-single LS {" ".join([str(s) for s in label_smoothing])}'

    arch = BERT(train.dna_encoder.vocab_size, d_model=256, d_ff=512, N=6)
    model = DeepITSClassifier(arch, train.dna_encoder, train.tax_encoder, 
                              output=output)

    loss = train.weighted_loss(CrossEntropyLoss, strength=0.5)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    try:
        model = DeepITSTrainer.train(model, train, valid, 50, loss,
            levels=[0,0,0,0,0,1], optimizer=optim, 
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
    
