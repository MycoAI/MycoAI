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
data = data.class_filter('species', min_samples=3) 
data = data.class_filter('genus', min_samples=3) 
data = data.class_filter('family', min_samples=3) 
data = data.class_filter('order', min_samples=3) 
data = data.class_filter('class', min_samples=3) 
data = data.class_filter('phylum', min_samples=3) 

data.labels_report()
data.unknown_labels_report()

plotter.counts_sunburstplot(data, id='filtered')
plotter.counts_boxplot(data, id='filtered')
plotter.counts_barchart(data, level='species', id='filtered')

train_data, valid_data = data.train_valid_split( 
    0.002, export_fasta=[f'/data/luuk/{name}.fasta', 
                         f'/data/luuk/{name}_valid.fasta']
)

# BPE tokenizer (for BERT)
dna_encoder = BytePairEncoder(train_data, length=256, vocab_size=768)
train = train_data.encode_dataset(dna_encoder, 
                                  export_path=f'/data/luuk/{name}_bpe.pt') 
valid_data.encode_dataset(dna_encoder, 
                          train.tax_encoder, 
                          export_path=f'/data/luuk/{name}_bpe_valid.pt')