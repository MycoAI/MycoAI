'''Creates a data subset that is used for training'''

from mycoai import utils
from mycoai.data import DataPrep
from mycoai.encoders import FourDimDNA, BytePairEncoder
import mycoai.plotter as plotter

utils.set_output_dir('subset', 'results')
data = DataPrep('/data/luuk/UNITE_public_25.07.2023_test_removed.fasta')
data = data.sequence_length_filter()
data = data.sequence_quality_filter()
data = data.class_filter('species', min_samples=5, max_classes=2500) 
plotter.counts_sunburstplot(data, id='filtered')
plotter.counts_boxplot(data, id='filtered')
plotter.counts_barchart(data, id='filtered')

dna_encoder = BytePairEncoder(data, length=512)
data.encode_dataset(dna_encoder, valid_split=0.2, 
                    export_path=['/data/luuk/subset_bpe.pt', 
                                 '/data/luuk/subset_bpe_valid.pt'])

dna_encoder = FourDimDNA()
data.encode_dataset(dna_encoder, valid_split=0.2, 
                    export_path=['/data/luuk/subset_4d.pt', 
                                 '/data/luuk/subset_4d_valid.pt'])