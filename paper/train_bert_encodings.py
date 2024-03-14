import torch
from mycoai import utils
from mycoai.train import SeqClassTrainer
from mycoai.data.encoders import KmerTokenizer, BytePairEncoder
from mycoai.train.loss import CrossEntropyLoss
from mycoai.modules import BERT, ResNet, SimpleCNN, SeqClassNetwork
from mycoai.data import Data, TensorData

utils.set_output_dir('/data1/s2592800')
utils.WANDB_PROJECT = 'Final Training'

smoothings = {'HLS': [0.02, 0.02, 0.02, 0.02, 0.02, 0],
              'SLS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
              'NoLS': [0.0, 0.0, 0.0, 0.0, 0.0, 0]}

valid_src = Data('/data1/s2592800/trainset_valid.fasta', allow_duplicates=True)
train_src = Data('/data1/s2592800/trainset.fasta', allow_duplicates=True) 

for k in [5,6]: # 4

    output = 'multi'
    lr = 0.0001
    wd = 0
    smoothing = 'HLS'
    arch = 'BERT-medium'

    name = f'MycoAI-{output}-{arch}-{k}mer-{smoothing}'

    output = 'infer_sum' if output=='single' else output
    dna_encoder = KmerTokenizer(k=k, length=256)
    train = train_src.encode_dataset(dna_encoder)
    valid = valid_src.encode_dataset(train.dna_encoder, train.tax_encoder)

    try:

        if arch == 'CNN-Vu-NoBN':
            arch = SimpleCNN(kernel=5,conv_layers=[5,10],in_channels=1, 
                             pool_size=2, batch_normalization=False)
            fcn_layers = [256]
        elif arch == 'CNN-Vu':
            arch = SimpleCNN(kernel=5,conv_layers=[5,10],in_channels=1, 
                             pool_size=2)
            fcn_layers = [256]
        elif arch == 'CNN-ResNet9':
            arch = ResNet([1,1,1,1])
            fcn_layers = [256]
        elif arch == 'CNN-ResNet18':
            arch = ResNet([2,2,2,2])
            fcn_layers = [256]
        elif arch == 'BERT-small':
            arch = BERT(train.dna_encoder.vocab_size, d_model=256, d_ff=512, 
                        N=6)
            fcn_layers = []
        elif arch == 'BERT-medium':
            arch = BERT(train.dna_encoder.vocab_size, d_model=512, d_ff=1024, 
                        N=8)
            fcn_layers = []
        elif arch == 'BERT-large':
            arch = BERT(train.dna_encoder.vocab_size, d_model=1024, d_ff=2048, 
                        N=10)
            fcn_layers = []
        else:
            raise ValueError("Invaild architecture.")

        model = SeqClassNetwork(arch, train.dna_encoder, train.tax_encoder, 
                                fcn_layers=fcn_layers, output=output)

        loss = train.weighted_loss(CrossEntropyLoss, strength=0.5)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        model = SeqClassTrainer.train(model, train, valid, 50, loss, 
            levels=[1,1,1,1,1,1], optimizer=optim, 
            label_smoothing=smoothings[smoothing], wandb_name=name)
        torch.save(model, f'/data1/s2592800/models/{name}.pt')

    except Exception as exception:
        print("Failed:", exception)