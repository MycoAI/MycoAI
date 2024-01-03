import torch
from mycoai import utils
from mycoai.deep.train import DeepITSTrainer
from mycoai.deep.train.loss import CrossEntropyLoss
from mycoai.deep.models import BERT, DeepITSClassifier
from mycoai.data import Data, TensorData
from mycoai.evaluate import Evaluator

utils.set_device('cuda:0') # TODO

valid_src = Data('/data/luuk/delete_valid.fasta', allow_duplicates=True)
train = TensorData(filepath='delete_bpe.pt')
valid = TensorData(filepath='delete_bpe_valid.pt')

label_smoothing = [0.033, 0.033, 0.033, 0, 0, 0]

for d_model in [256, 512]:
    for d_ff in [d_model*2, d_model*4]:
        for N in [6, 8, 10]:

            arch = BERT(train.dna_encoder.vocab_size, d_model, d_ff, 8, N)
            model = DeepITSClassifier(arch, train.dna_encoder, train.tax_encoder, 
                                    output='multi')

            # TODO see how fast one epoch is
            loss = train.weighted_loss(CrossEntropyLoss)
            optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
            try:
                model, history = DeepITSTrainer.train(model, train, valid, 40, loss, 
                    optimizer=optim, label_smoothing=label_smoothing, levels=[1,1,1,1,1,0], 
                    wandb_name=f'd_m={d_model}, d_ff={d_ff}, N={N}')
                
                classification = model.classify(valid)
                latent_repr = model.latent_space(valid)
                evaluator = Evaluator(classification, valid_src, model, 
                            wandb_name=f'd_m={d_model}, d_ff={d_ff}, N={N} (test)')
                evaluator.test()
                for level in utils.LEVELS:
                    evaluator.detailed_report(level, latent_repr=latent_repr)
            except Exception as exception:
                print("Failed:", exception)
    
