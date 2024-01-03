'''Encodes the full UNITE dataset with BPE'''

from mycoai import utils
from mycoai.data import Data
from mycoai.data.encoders import BytePairEncoder
import mycoai.plotter as plotter

name = 'trainset'

utils.set_output_dir(name, 'results')
data = Data('/data/luuk/UNITE_public_25.07.2023_test_removed.fasta')
data = data.sequence_length_filter()
data = data.sequence_quality_filter() 
data = data.class_filter('genus', min_samples=5, remove_unidentified=True) 
data = data.remove_level_filter('species')
plotter.counts_sunburstplot(data, id='filtered')
plotter.counts_boxplot(data, id='filtered')
plotter.counts_barchart(data, id='filtered')

train_data, valid_data = data.train_valid_split( 
    0.002, export_fasta=[f'{name}.fasta', 
                         f'{name}_valid.fasta']
)

dna_encoder = BytePairEncoder(data, length=256, vocab_size=768)
train = train_data.encode_dataset(dna_encoder, 
                                  export_path=f'{name}_bpe.pt') 
valid_data.encode_dataset(dna_encoder, 
                          train.tax_encoder, 
                          export_path=f'{name}_bpe_valid.pt')