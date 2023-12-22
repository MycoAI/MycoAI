'''Creates a data subset that is used for training'''

from mycoai import utils
from mycoai.data import Data
from mycoai.data.encoders import BytePairEncoder
import mycoai.plotter as plotter

utils.set_output_dir('subset', 'results')
data = Data('/data/luuk/UNITE_public_25.07.2023_test_removed.fasta')
data = data.class_filter('species', min_samples=5, max_classes=3000) 
data = data.sequence_length_filter()
data = data.sequence_quality_filter()
plotter.counts_sunburstplot(data, id='filtered')
plotter.counts_boxplot(data, id='filtered')
plotter.counts_barchart(data, id='filtered')

train_data, valid_data = data.train_valid_split(
    0.1, export_fasta=['/data/luuk/subset.fasta',
                       '/data/luuk/subset_valid.fasta']
)

dna_encoder = BytePairEncoder(data, length=256, vocab_size=768)
train = train_data.encode_dataset(dna_encoder, 
                                  export_path='/data/luuk/subset_bpe.pt') 
valid_data.encode_dataset(dna_encoder, 
                          train.tax_encoder, 
                          export_path='/data/luuk/subset_bpe_valid.pt')