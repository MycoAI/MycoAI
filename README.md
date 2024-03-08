# About MycoAI
Welcome! MycoAI is a Python package that implements deep learning models for the
classification of (fungal ITS) barcode sequences. Our most sophisticated model
is trained on the [UNITE](https://unite.ut.ee/) dataset, giving it a class 
coverage of 14,742 species and 3,695 genera. In our [paper](todo.com), we show 
that MycoAI achieves a higher accuracy than RDP classifier (the previous 
state-of-the-art fast fungal classification algorithm) on independent test sets. 
Furthermore, MycoAI is capable of classifying large datasets of 100,000+ 
sequences in <5 minutes, making it a very suitable algorithm for large-scale 
biodiversity studies. We refer to 
[DNABarcoder](https://github.com/MycoAI/dnabarcoder) for the most precise (and 
most computationally demanding) classification.

In short, the package allows users to:
- **Classify** their own fungal ITS datasets.
- **Train** neural networks for the taxonomic classification of biological 
sequences.

## Contact
Questions can be directed to 
[Luuk Romeijn](mailto:l.romeijn@umail.leidenuniv.nl).

# Installation
Currently, the only way of using MycoAI is from source:
    
    git clone https://github.com/MycoAI/MycoAI

You can install the specified requirements manually, or create a conda 
environment with all the necessary dependencies using the command below. 

    conda env create -f environment.yml

## Requirements
MycoAI requires the following packages: [Python]((https://www.python.org/)) 
(version 3.8 or higher), [Biopython](https://biopython.org/)
, [Pandas](https://pandas.pydata.org/)
, [Plotly](https://plotly.com/python/)
, [PyTorch](https://pytorch.org/)
, [Matplotlib](https://matplotlib.org/)
, [Numpy](https://numpy.org/)
, [Scikit-learn](https://scikit-learn.org/)
, [Scipy](https://scipy.org/)
, [SentencePiece](https://github.com/google/sentencepiece)
, [Tqdm](https://github.com/tqdm/tqdm)
, and [Weights and Biases](https://wandb.ai/site). The least cumbersome
installation process uses the environment file (explained 
[above](#installation)).

A Graphical Processing Unit (GPU) is not required but will speed up training and
classification significantly. 

# Getting started
We provide single-command scripts to:
- [Classify](#classification) your own fungal ITS dataset.
- [Evaluate](#evaluation) MycoAI's performance with your own labelled 
dataset.
- [Map](#taxonomic-mapping) your own ITS sequences onto MycoAI's taxonomic 
map.
- [Train](#training) a new neural network with your own dataset.

An example of how to use the scripts is given 
[here](https://api.wandb.ai/links/mycoai/16ap85ig). 

If you find the functionalitiy of these scripts to be too limited for your
needs, we highly encourage you to write your own scripts using the available 
modules. MycoAI was made to be modular and easily integrated into other 
projects. The package is documented [below](#the-mycoai-package).

## Weights and Biases
Your first run of MycoAI will prompt the following Weights and Biases 
login option:

    wandb: (1) Private W&B dashboard, no account required
    wandb: (2) Create a W&B account
    wandb: (3) Use an existing W&B account
    wandb: (4) Don't visualize my results
    wandb: Enter your choice: 

Here, choosing option 1 is sufficient. Weigths and Biases is an AI development
platform, which MycoAI mainly uses for live visualization of results. 

# Classification

    python -m scripts.classify <fasta_filepath>

Taxonomic classification of fungal ITS sequences using a deep neural network.
Output is written to a .csv file.

| Argument | Required | Description | Values | 
| ---  | --- | --- | --- |
| `fasta_filepath` | Yes | Path to the FASTA file containing ITS sequences. | path 
| `--out`   | No | Where to save the output to (default is 'prediction.csv') | path
| `--model` | No |  Path to saved SeqClassNetwork model (default is 'models/MycoAI-multi-HLS.pt'). | path
| `--device` | No | Forces use of specific device. By default, MycoAI will look for and use GPUs whenever available and falls back to CPU if unsuccessful. | ['cpu', 'cuda', 'cuda:0', etc.]

# Evaluation

    python -m scripts.evaluate <classification> <reference>

Evaluates predicted classification of fungal ITS sequences. Results will be 
printed, graphically displayed on Weights and Biases, and written to a .csv
file. 

| Argument | Required | Description | Values | 
| ---  | --- | --- | --- |
| `classification` | Yes | Path to .csv file containing predicted labels. | path 
| `reference` | Yes | Path to .csv or FASTA file containing ground truth labels. | path
| `--out` | No | Where to save the output to (default is 'evaluate.csv') | path

# Taxonomic mapping

    python -m scripts.map <fasta_filepath>

Creates a 2D visualization of the final layer of a ITS sequence classification 
network, serving as a taxonomic map. Output is an .html file which can be
opened in your favourite browser. The hidden representation of the final layer of the network is reduced via the t-distributed Stochastic Neighbourhood Embedding 
(t-SNE) algorithm.

The sequences in the provided FASTA file
will be mapped together with a background dataset to serve as context. By
default, a randomly selected subset of the UNITE datset is used as context.
Should you wish to investigate how your 'unknown' sequences relate to a set of
labelled sequences, then provide this labelled dataset in the `--context`
argument. 

| Argument | Required | Description | Values | 
| --- | --- | --- | --- |
| `fasta_filepath` | Yes | Path to the FASTA file containing ITS sequences. | path |
| `--out` | No | Where to save the output to (default is 'map.html'). | path | 
| `--classification` | No | Classification of input sequences in .csv (default is None). | path 
| `--context` | No | Set of labelled sequences, provided in FASTA format, to serve as context in the taxonomic map. (default is 'data/trainset_valid.fasta'). | path
| `--model` | No | Path to saved SeqClassNetwork Pytorch model (default is 'models/MycoAI-multi-HLS.pt'). | path
| `--device` | No | Forces use of specific device. By default, MycoAI will look for and use GPUs whenever available and falls back to CPU if unsuccessful. | ['cpu', 'cuda', 'cuda:0', etc.]

# Training
    
    python -m scripts.train <fasta_filepath>

Trains a deep neural network for taxonomic classification of fungal ITS 
sequences. Output is a SeqClassNetwork object stored in a .pt file. Training
progress can be followed live (updated every epoch) via Weights and Biases. 

Even though the script contains many optional arguments for customization, the 
full functionality of MycoAI can only be accessed through custom-made scripts 
that users can write themselves with the [MycoAI package](#the-mycoai-package).

| Argument | Required | Description | Values | 
| ---  | --- | --- | --- |
| `fasta_filepath` | Yes | Path to the FASTA file containing ITS sequences for training. | path |
| `--out` | No | Path to where the trained model should be saved to (default is 'model.pt'). | path | 
| `--base_arch_type` | No |  Type of the to-be-trained base architecture (default is BERT). | ['CNN', 'BERT'] | 
| `--output_type` | No |  Whether to use the multi-head output or not (default is 'multi'). | ['multi', 'infer_sum']
| `--no_hls` | No | If specified, turns off hierarchical label smoothing (default is HLS) | -
| `--levels` | No | Specifies the levels that should be trained (or their weights). Must be space-separated. Can be a list of strings, e.g. 'family genus'. Can also be a list of floats, indicating the weight per level, e.g. '0 0 0 1 1 0' (default is al levels, i.e. '1 1 1 1 1 1'). | list of strings, floats, or integers (space-separated)  
| `--epochs` | No | Number of epochs to train for (default is 50). | `int` | 
| `--batch_size` | No | Number of samples per batch (default is 64). | `int` | 
| `--valid_split` | No | Fraction of data to be used for validation (default is 0.1). | `float` | 
| `--learning_rate` | No | Learning rate in Adam optimizer (default is 0.0001). | `float` | 
| `--weighted_loss` | No | Strength of how much per-class-loss should be weighted by the reciprocal class size, for imbalanced classes (default is 0.5). | `float`
| `--device` | No | Forces use of specific device. By default, MycoAI will look for and use GPUs whenever available and falls back to CPU if unsuccessful. | ['cpu', 'cuda', 'cuda:0', etc.]

# The MycoAI package
MycoAI is a deep learning based (fungal ITS) taxonomic sequence classification 
development platform built upon the PyTorch framework. It was designed to be modular and easily integrated into 
other projects. This is achieved through an object-oriented design. The most 
important modules are: 
* [`mycoai.data.Data`](#mycoaidatadata): data parsing, filtering, and encoding to `TensorData` 
object.
* [`mycoai.data.encoders`](#mycoaidataencoders): contains several encoding methods that aid in 
converting a `Data` object to a network-readable `TensorData` object.
* [`mycoai.data.TensorData`](#mycoaidatatensordata): network-readable data format (Tensors).
* [`mycoai.modules.BERT`](#mycoaimodulesbert): transformer-based architecture. 
* [`mycoai.modules.SeqClassNetwork`](#mycoaimodulesseqclassnetwork): wrapper class that stores encoder objects 
and adds an output layer as well as other application-related functionalities to 
a base architecture like `BERT`.
* [`mycoai.train.SeqClassTrainer`](#mycoaitrainseqclasstrainer): trains a `SeqClassNetwork` model.
* [`mycoai.evaluate.Evaluator`](#mycoaievaluateevaluator): evaluates a classification output.
* [`mycoai.utils`](#mycoaiutils): contains constants and helper functions.

An example script is available [here](example.py). This script should be able to
run on your machine without any modifications directly after installing MycoAI. 

## `mycoai.data.Data`
Data container that parses and filters sequence data and encodes it into the 
network-readable `TensorData` format. After initializing a `Data` object, the
sequence/taxonomy data is stored within its `data` attribute. 

* `__init___(self, filepath, tax_parser='unite', allow_duplicates=False, name=None)`
    * `filepath`: Path of to-be-loaded file in FASTA format (`str`)
    * `tax_parser`: Function that parses the FASTA headers and extracts the taxonomic labels on all six levels (from phylum to species). If `None`, no labels will be extracted. If 'unite', will follow the UNITE format. Also supports user-custom functions, as long as they return a list following the format: [id, phylum, class, order, family, genus, species] (function, default is 'unite').
    * `allow_duplicates`: Drops duplicate entries if `False` (default is `False`)
    * `name`: Name of dataset. Will be inferred from filename if `None` (default is
    `None`).

### Data parsing
By default, the UNITE labelling format is assumed:

    >KY106084|k__Fungi;p__Ascomycota;c__Saccharomycetes;o__Saccharomycetales;f__Saccharomycetaceae;g__Zygotorulaspora;s__Zygotorulaspora_florentina|SH0987707.09FU

If your dataset deviates from this format, you can write your own parsing
function and pass it into the `tax_parser` argument. Any function is accepted,
as long as it returns a list of the format [id, phylum, class, order, family, 
genus, species]:

```python
from mycoai.data import Data

def custom_parser(fasta_header):
    '''Example parsing function for header with comma-separated labels'''
    return = fasta_header.split(",")

dataset = Data('dataset.fasta', tax_parser=custom_parser)
```

### Data filtering
The `Data` object contains several methods for data filtering and manipulation, 
such as:

* `train_valid_split(self, valid_split, export_fasta=False)`: splits up the 
`Data` object into a train and validation split, returning two `Data` objects.
Writes FASTA file if `export_fasta` is `True`. If `export_fasta` is of type 
`list[str]`, will write train and validation dataset to specified filepaths.

* `class_filter(self, level, min_samples=0, max_samples=np.inf, max_classes=np.inf, remove_unidentified=False)`: This method can be used for reducing class 
imbalance by filtering out sequences as well as for creating manageable data 
subsets. Retains at most `max_samples` sequences at specified taxon `level` for 
which at least `min_samples` are available in that class. Ensures perfect class 
balance when `min_samples==max_samples`. Randomly selects a `max_classes` number
of classes.

* `sequence_quality_filter(self, tolerance=0.05)`: Removes sequences with more 
than tolerated number of uncertain bases.

* `sequence_length_filter(self, tolerance=4)`: Removes sequences with that fall 
outside of the tolerated range.
    * `tolerance`: Tolerated range of lengths. In case of an integer, the tolerated range is defined as the sequences that fall within the specified number of standard deviations from the mean length (`int|list|range`, default is 4).

* `export_fasta(self, filepath)`: Exports this `Data` object to a FASTA file.

Data filtering operations happen in-place even though each of the filtering 
methods also return the filtered object (`self`).

```python
from mycoai.data import Data

# Load data
data = mycoai.Data('dataset.fasta') 
# Select a subset of 1000 species from those that have at least 5 samples
data = data.class_filter('species', min_samples=5, max_classes=1000)
# Remove sequences with more than 5% of bases not in [A,C,G,T]
data = data.sequence_quality_filter(tolerance=0.05)
# Remove sequences with more than 4 stds from the mean length
data = data.sequence_length_filter(tolerance=4)
# Export to FASTA file
data.export_fasta('filtered.fasta')
```

### Data encoding
The data must be encoded (i.e. converted into tensors of numbers) before being 
inputted into a neural network. The following method converts a `Data` object
into a `TensorData` object that our sequence classifiers can operate on:

* `encode_dataset(self, dna_encoder, tax_encoder='categorical', export_path=None)`: Converts data into a `TensorData` object. 
    * `dna_encoder`: Specifies the encoder used for generating the sequence tensor. Can be an existing object, or one of ['4d', 'kmer-tokens', 'kmer-onehot', 'kmer-spectral', 'bpe'], which will initialize an encoder of that type (`DNAEncoder|str`).
    * `tax_encoder`: Specifies the encoder used for generating the taxonomies tensor.
    Can be an existing object, or 'categorical', which will initialize an encoder of that type (`TaxonEncoder`, default is 'categorical').
    * `export_path`: Path to save encodings to (`str`, default is `None`).

More information about the encoders and their compatibility with several
network types is given [here](#mycoaidataencoders). 

**NOTE**: When using a training/validation split, it is very important to encode the validation dataset with the same encoders that were used for the training dataset (such that e.g. Ascomycota is assigned to the same index in both datasets). This can be achieved by passing the `dna_encoder` and `tax_encoder` attributes of the newly created `TensorData` object into the `dna_encoder` and `tax_encoder` arguments of the `Data.encode_dataset` method:

```python
from mycoai.data import Data

# Load data and split into training and validation set
data = Data('dataset.fasta')
train_data, valid_data = data.train_valid_split(0.2)

# It is important to encode the validation dataset with the same encoders
train_data = train_data.encode_dataset('bpe') # Byte Pair Encoding
valid_data = valid_data.encode_dataset(dna_encoder=train_data.dna_encoder,
                                       tax_encoder=train_data.tax_encoder)

```

## `mycoai.data.encoders`
Contains several encoding methods that aid in converting a `Data` object to a network-readable `TensorData` object. These encoding methods are very important:
not only are they used for encoding the training/validation datasets, they must
also be stored *in* the sequence classification network such that it can encode 
any future dataset by itself and decode its own predictions. Two encoding
schemes are required, one for encoding the DNA sequences and one for
en-/decoding taxonomic labels.

### Encoding DNA sequences
MycoAI includes several DNA encoding techniques, each compatible with specific 
neural architecture designs. They all inherit from the same `DNAEncoder` base 
class. An overview is given below:

| Name | Description | Tensor shape* | Example encoding: ACGACGT | Model |
| --- | --- | --- | --- | --- |
| `BytePairEncoder` | Keeps track of the most frequently appearing combinations of characters, resulting in a fixed-sized vocabulary of flexibly-sized words. The occurrence of a word is indicated by a unique index (token). | $[n,l]$ | `[[1,5,6,2]]` assuming tokens 'ACGA' and 'CGT' | Transformer
| `KmerSpectrum` | Encodes each sequence into a frequency vector for its $k$-mers. | $[n,1,4^k]$ | `[[[0.12,...,0.07,...]]]` | CNN
| `FourDimDNA` | 4-channel representation, comparable to the 3-channel RGB representation of images. | $[n,4,l]$ | `[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]` | CNN
| `KmerTokenizer` | Encodes each possible $k$-mer with a unique index (token) | $[n,l]$ | `[[1,5,5,2]]` for $k=3$ | Transformer
| `KmerOneHot` | One-hot encoding of $k$-mers, given each possible $k$-mer its own channel | $[n,4^k,l]$ | `[[[0,...,1,...,0],[0,...,1,...,0]]]` | CNN

**=for $n$ sequences with (padded) length $l$*

Note that in case of `BytePairEncoder` and `KmerTokenizer`, five token values
have special meanings. For example, in the table above 1 indicates the CLS 
(start) token and 2 indicates the SEP (end) token. We also use dedicated tokens 
for padding, unknown values, and masking. 

The `BytePairEncoder` is currently the only DNAEncoder that has a `encode_fast`
method implemented. This method allows the encoder to efficiently translate an
entire `Data` object into a tensor, making it (by far) the fastest encoding 
technique. 

The example below shows how to initialize the most important encoding methods 
with compatible architectures.
```python
from mycoai.data import Data
from mycoai.encoders import *
from mycoai.modules import BERT, SimpleCNN

data = Data('dataset.fasta')

# Byte Pair Encoding with a transformer
dna_encoder = BytePairEncoder(data) # Must initialize on Data object
arch = BERT(dna_encoder.vocab_size) # Must specify vocab_size

# K-mer encoding with transformer
dna_encoder = KmerTokenizer(k=4)
arch = BERT(dna_encoder.vocab_size) # Again, must specify vocab_size

# K-mer spectral encoding with CNN
dna_encoder = KmerSpectrum(k=4)
arch = SimpleCNN(in_channels=1) # 1 input channel

# 4D encoding with CNN
dna_encoder = FourDimDNA()
arch = SimpleCNN(in_channels=4) # 4 input channels

# (Can also do this before initializing architecture:)
data = data.encode_dataset(dna_encoder)
```

### Encoding taxonomic labels
`TaxonEncoder`: Sparse categorical encoding method of taxonomic labels on 6 levels. For every level, each label is assigned to a unique index by a list of encoders stored in its `lvl_encoders` attribute. Furthermore, hierarchical `inference_matrices` are generated during training, as the model keeps track of which parent label a sequence of a specific child label is most oftenly assigned to.

* `__init__(self, data)`: A `TaxonEncoder` object requires initialization with a `Data` object passed as argument. This will initialize the label encoders as well as the inference matrices.
* `encode(self, data_row)`: Assigns integers to taxonomic level. Also builds inference matrices during training, for which it assumes that this method is called for all rows of the training dataset.
* `encode_fast(self, data)`: Assigns integers to all taxonomic levels, optimized to operate on an entire `Data` object at once (use only outside of training).
* `decode(self, labels: np.ndarray, levels: list=utils.LEVELS)`: Decodes an array of index `labels` into their corresponding strings at specified `levels`.
* `infer_parent_probs(self, y, parent_lvl)`: Calculates probabilities for parents given child probabilities based on inference matrices (Equation 1 in paper).
* `infer_child_probs(self, y, child_lvl)`: Calculates probabilities for children given parent probabilities, based on inference matrices (Equation 3 in paper).

## `mycoai.data.TensorData`
Holds data as tensors in `sequences` and `taxonomies` attributes. The class 
directly inherits from `torch.utils.data.Dataset`, allowing it to work with 
PyTorch dataloaders. A `TensorData` object also contains the exact encoder 
instances that were used for generating the tensors in the dataset, stored in 
its `dna_encoder` and `tax_encoder` attributes.

* ` __init__(self, sequences=None, taxonomies=None, dna_encoder=None,tax_encoder=None, name=None, filepath=None)`: Initializes TensorData object from tensors (`Data.encode_dataset` will use this), or imports a pre-saved object from the specified filepath (after being exported by `TensorData.export_data`).

* `export_data(self, export_path)`: Saves sequences, taxonomies, and encoders to file. This is useful when your dataset is very large and encoding is slow.

Other built-in `TensorData` methods provide insight into the class distribution
and/or are related to weighted loss/sampling.

### Weighted loss/sampling
The `TensorData` contains two methods that are related to counteracting the 
class imbalance problem of many taxonomically labelled datasets.

* `weighted_sampler(self, level='species', strength=1.0, unknown_frac=0.0)`: Yields a random sampler that balances out (either fully or to some extent) the label distributions by over-/undersampling small/large classes. 
    * `level`: Level at which the data balancing will be applied to (`str`).
    * `strength`: Amount of balancing to be applied. If 0, maintains the original data distribution. If 1, ensures perfect class balance. Numbers in between represent varying degrees of balance, where the weight of a class of size 
    $c$ is determined by ${c^{-strength}}$ (`float`, default is 1.0). 
    * `unknown_frac`: Sample unidentified classes an `unknown_frac` fraction of times. Will not sample the unidentified class if 0, potentially leading to a
    lot of wasted data (`float`, default is 0.0).

* `weighted_loss(self, loss_function, sampler=None, strength=1.0)`: Returns a list of six weighted loss functions that balance out (either fully or to some extent)
the label distributions by weighting small/large classes more/less. 
    * `loss_function`: Loss function to be used as basis. The recommended loss function is `mycoai.train.CrossEntropyLoss`. Loss function must support the
    specification of a `weight` and `ignore_index` parameter 
    (`torch.nn.Module`).
    * `sampler`: If provided, will correct for the expected data distribution (on all taxonomic levels) given the specified sampler. Thus, you can use both a 
    weighted sampler ánd a weighted loss function with varying strengths.
    * `strength`:  Amount of balancing to be applied. If 0, maintains the original data distribution. If 1, ensures perfect class balance. Numbers in between represent varying degrees of balance, where the weight of a class of size 
    $c$ is determined by ${c^{-strength}}$ (`float`, default is 1.0). 

Example usage of the `TensorData` object is provided below.
```python
import torch
from mycoai.data import Data, TensorData
from mycoai.train import CrossEntropyLoss, SeqClassTrainer

# One way of creating a TensorData object (encoding a Data object)
data = Data('dataset.fasta').encode_dataset('bpe')

# Exporting the newly created object
data.export_data('dataset.pt')

# Alternative way of creating TensorData object (loading stored object)
data = TensorData(filepath='dataset.pt')

# Applying weighted sampling and weighted loss
sampler = data.weighted_sampler(strength=0.5) # Not fully balanced
# Remaining class imbalance is counteracted using weighted loss
loss = data.weighted_loss(CrossEntropyLoss, strength=1, sampler=sampler)

# Weighted loss/sampler can later be inputted to training procedure
model = torch.load('model.pt')
model, history = SeqClassTrainer.train(model, data, sampler=sampler, loss=loss)
```

## `mycoai.modules.BERT`
BERT base model, transformer encoder to be used for various tasks. Inherits from
`torch.nn.Module`, just like all other pre-implemented architectures that are
part of MycoAI. 
* `__init__(self, vocab_size, d_model=256, d_ff=512, h=8, N=6, dropout=0.1, mode='default')`: initializes the transformer given the specified
hyperparameters and vocabulary size.
    * `vocab_size`: Number of unique tokens in vocabulary. Can be the `vocab_size` attribute of `BytePairEncoder` or `KmerTokenizer` (`int`). 
    * `d_model`: Dimension of token representations (embeddings) in model (`int`, default is 256)
    * `d_ff`: Dimension of hidden layer feed-forward sublayers (`int`, default is 512)
    * `h`: Number of heads used for multi-head self-attention, must be a divisor of `d_model` (`int`, default is `8`).
    * `N`: How many encoder/decoder layers the transformer has (`int`, default is 6)
    * `dropout` Dropout probability to use throughout network (`float`, default is 0.1)
    * `mode`: Determines the forward method that BERT will use. Users are never required to specify this, MycoAI will change the mode of BERT depending on the task (One of ['default', 'classification', 'mlm']).

Note that MycoAI supports the pre-training of a `BERT` module through the `mycoai.train.MLMTrainer` class. This will perform Masked Language Modelling on the sequences that are part of the specified dataset.

```python
import torch
from mycoai.data import TensorData
from mycoai.modules import BERT
from mycoai.train import MLMTrainer

data = TensorData(filepath='dataset.pt') # Reading TensorData from file

# Initializing BERT module
model = BERT(vocab_size=data.dna_encoder.vocab_size)

# Pre-training BERT
model, history = MLMTrainer.train(model, data, epochs=50)
torch.save(model, 'pretrained.pt') # Saving the pre-trained model
```

## `mycoai.modules.SeqClassNetwork`
Performs taxonomic classification based on DNA sequences. It is a wrapper class
that stores encoder objects and adds an output layer as well as other 
application-related functionalities to a base architecture (such as `BERT`).
The encoders are stored in its `dna_encoder` and `tax_encoder` attributes. 
The base architecture and output network are found in its `base_arch` and 
`output` attributes, respectively. Is stored on `utils.DEVICE` upon 
initialization.

* `__init__(self, base_arch, dna_encoder, tax_encoder, fcn_layers=[], dropout=0, output='multi', max_level='species', chained_config=[False,True,True])`: Initializes the module for the specified base architecture, encoders, and output head.
    * `base_arch`: The body for the neural network. Can be one of the pre-implemented MycoAI modules as well as any custom PyTorch module (`torch.nn.Module`).
    * `dna_encoder`: The DNA encoder used for the expected input (`DNAEncoder`).
    * `tax_encoder`: The label encoder used for the (predicted) labels (`TaxonEncoder`).
    * `fcn_layers`: List of node numbers for fully connected part before the output head (`list[int]`, default is `[]`).
    * `dropout`: Dropout percentage for the dropout layer (`float`, default is 0).
    * `output`: The type of output head(s) for the neural network. More information is found [here](#output-architectures) (One of ['infer_parent', 'infer_sum', 'multi', 'chained', 'tree'], default is 'multi').
    * `max_level`: Until what level to predict (only for 'infer_parent', 'infer_sum', and 'multi' output heads) (`str`, default is 'species').
    * `chained_config`: List of length 3 indicating the configuration for `ChainedMultiHead`. Corresponding to arguments: `ascending`, `use_probs`, and `all_access` (`list[bool]`, default is [`False`, `True`, `True`]).

* `__call__(self, x)`: A forward pass through the neural network. Outputs a list of six tensors, one per taxonomic level (from phylum to species). Calls the `forward` method. Note that the `forward` method has several alternative implementations to support extracting the latent space as well autoregressive models (with/without teacher forcing). 

* `classify(input_data)`: Classifies sequences in FASTA file, `Data`, or `TensorData` object, returns a pandas `DataFrame`. If required, will parse/encode 
`input_data` before classification. 

* `latent_space(input_data)`: Extracts latent space for given input data. Equals a forward pass up until the final layer of the base architecture or the bottleneck (least no. dimension) layer of the fully-connected component of the network (e.g. in case of a CNN). Input data can be a path to a FASTA file, or a `Data` or `TensorData` object. Returns a numpy array. If required, will parse/encode `input_data` before classification. 

### Output architectures
An important component of a `SeqClassNetwork` object is its output architecture. 
Implementations for several (experimental) output architectures are given in 
`mycoai.modules.output_heads`. If you wish your own output architecture to be supported by MycoAI's models and its training procedure, make sure to output a list 
of 6 tensors, each corresponding to a taxonomic level (from phylum to species). The most relevant output architectures are explained below. 

| Name | Description |
| --- | --- |
| `MultiHead`| Predicting multiple taxon levels using different and independent output heads (softmax-activated linear layers). This output architecture was proven to yield models with better representations of the (hierarchical) taxonomic space and is the recommend output architecture.
| `ChainedMultiHead` | Like `MultiHead`, but each taxon level also gets input from the previously predicted level. Whether these connections move from phylum-to-species or species-to-phylum is determined by `ascending=False` or `True`, respectively. The `all_access` argument defines whether the next output head is also inputted with the base architecture output. The `use_prob` argument determines whether probabilities are forwarded to the next level or the original, non-softmax-activated linear output. 
| `InferSum`| Sums Softmax probabilities of child classes to infer the probability of a parent, using the taxon encoder's inference matrices. This allows a single-headed output architecture while still classifying all taxonomic levels.
| `InferParent` | Like `InferSum`, but infers parent classes by looking in the inference matrix and seeing what parent a child class is most often part of (instead of summing). 

Note that a `MultiHead` output can be converted to an `InferSum` output head by calling the `multi_to_infer_sum` method of a `SeqClassNetwork` object.

### Storing/loading a model
A `SeqClassNetwork` object can be stored for later use via `torch.save` and loaded through `torch.load`:

```python
import torch
from mycoai.data import TensorData
from mycoai.modules import BERT, SeqClassNetwork

data = TensorData(filepath='dataset.pt') # Reading TensorData from file
arch = BERT(data.dna_encoder.vocab_size) # Initializing BERT module

# Creating SeqClassNetwork model and saving it
model = SeqClassNetwork(arch, data.dna_encoder, data.tax_encoder)
torch.save(model, 'model.pt')

# Loading an existent one (either on CPU/GPU via utils.DEVICE)
mycoai_trained = torch.load('models/MycoAI-multi-HLS.pt', 
                            map_location=utils.DEVICE)

# Making and saving a classification
prediction = mycoai_trained.classify(data)
prediction.to_csv('prediction.csv') 
```

## `mycoai.train.SeqClassTrainer`
Trains a `SeqClassNetwork` object for multi-class classification on 6 taxonomic 
levels. Requires no initialization, i.e. all methods are static methods. 
Supports Hierarchical Label Smoothing (HLS), mixed precision, and curriculum 
learning. Will initialize a Weights and Biases run via which several metrics
can be tracked in real time, which is updated every training epoch.

* `train(model, train_data, valid_data=None, epochs=100, loss=None, batch_size=64, sampler=None, optimizer=None, metrics=utils.EVAL_METRICS, levels=utils.LEVELS, warmup_steps=None, label_smoothing=[0.02,0.02,0.02,0.02,0.02,0], wandb_config={}, wandb_name=None)`: Trains a `SeqClassNetwork` object to taxonomically classify sequences. Returns a tuple in which the first returned element corresponds to 
the trained model, and the second element corresponds to a history dataframe
that contains the performance per epoch.

    * `model`: Neural network architecture (`torch.nn.Module`).
    * `train_data`: Preprocessed dataset containing ITS sequences for training (`mycoai.data.TensorData`).
    * `valid_data`: Preprocessed dataset containing ITS sequences for validation (`mycoai.data.TensorData`).  
    * `epochs`: Number of training iterations (`int`, default is 50).
    * `loss`:To-be-optimized loss function (or list of functions per level) 
        (`list` | `function`, default is `CrossEntropyLoss`).
    * `batch_size`: Number of training examples per optimization step (`int`, default is 64)
    * `sampler`: Strategy to use for drawing data samples (`torch.utils.data.Sampler`)
    * `optimizer`: Optimization strategy (default is Adam) (`torch.optim`)
    * `metrics`: Evaluation metrics to report during training, provided as dictionary
    with metric name as key and function as value (`dict{str:function}`, default is
    accuracy, balanced acuracy, precision, recall, f1, and mcc).
    * `levels`: Specifies the levels that should be trained (and their weights).
    Can be a list of strings, e.g. ['genus', 'species]. Can also be a list of floats, indicating the weight per level, e.g. [0,0,0,0,1,1]. Can also be a MycoAI weight schedule object, e.g. `Constant([0,0,0,0,1,1])` (`list`| `mycoai.train.weight_schedules`, default is `utils.LEVELS`).
    * `warmup_steps`: When specified, the lr increases linearly for the first warmup_steps then decreases proportionally to 1/sqrt(step_number). Works only for models with d_model attribute (BERT/EncoderDecoder) (`int`|`NoneType`,default is 0).
    * `label_smoothing`: Explained [here](#using-hierarchical-label-smoothing-hls). List of six decimals that controls how much label smoothing should be added per taxonomic level. The sixth element of this list refersto the amount of weight that is divided uniformly over all classes. Hence, [0,0,0,0,0,0.1] corresponds to standard label smoothing with $\epsilon=0.1$, whereas [0.02,02,0.02,0.02,0.02,0] corresponds to hierarchical label smoothing with $\epsilon=0.1$ (`list[float]`, default is [0.02,0.02,0.02,0.02,0.02,0]).
    * `wandb_config`: Extra information to be added to weights and biases config data (`dict`).
    * `wandb_name`: Name of the run to be displayed on weights and biases (`str`).

### Using Hierarchical Label Smoothing (HLS)
By default, the `SeqClassTrainer.train` method will use Hierarchical Label 
Smoothing (HLS). HLS is a method that, like label smoothing, transforms the 
original one-hot target distribution into a soft distribution that does not only
put weight on the target class but also on other classes. The difference with 
standard label smoothing is that the amount of weight per class is not uniformly
distributed, but determined by the target labels on higher taxonomic levels. 
Whether or not a child class is part of the higher-level target label determines
whether extra weight is added to this class. In other words, **the amount of 
label smoothing for a certain taxon is determined by how hierarchically similar 
this taxon is to the true target label**. For instance, if the true label at 
family-level is Agaricaceae, then a certain amount of weight is added to all 
genera and species that are part of the Agaricaceae family. This is explained 
more formally in our paper. 

The amount of HLS is controlled by the `label_smoothing` argument of the
`SeqClassTrainer.train` method. It is a list of six decimals that controls how 
much label smoothing should be added per taxonomic level. Specifically, the 
values represent how much weight is divided over child classes of correct parent 
labels. E.g. if `label_smoothing[0]==0.1` and the target phylum P1 has two child
taxons C1 and C2, both of these taxons will receive an 0.1/2 amount of weight 
added to their target distributions. Note that the sixth element of this list 
does not correspond to species-level smoothing (as species don't have child 
classes) but refers to the amount of weight that is divided uniformly over all 
classes. Hence, [0,0,0,0,0,0.1] corresponds to standard label smoothing with 
$\epsilon=0.1$, [0.02,02,0.02,0.02,0.02,0] to hierarchical label smoothing with 
$\epsilon=0.1$ 

```python
import torch
import torch
from mycoai.data import TensorData
from mycoai.train import SeqClassTrainer

train = TensorData(filepath='train_dataset.pt')
valid = TensorData(filepath='validation_dataset.pt')
model = torch.load('model.pt')

settings = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.1], # Standard Label Smoothing
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.0], # Hierarchical Label Smoothing
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.0], # No Label Smoothing
    None, # No Label Smoothing (alternative)
]

# E.g. custom optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for smoothing in settings:
    model, history = SeqClassTrainer.train(
        model, train, valid, label_smoothing=smoothing, optimizer=optimizer
    )
```

## `mycoai.evaluate.Evaluator`
Evaluation and analysis of ITS classification algorithms. An `Evaluator` object
is initialized for a specific classifier-dataset combination. It is designed to 
operate on classifications (in `pd.DataFrame` format) made by *any* type of 
classifier, also non-deep learning classifiers. Results will be graphically 
displayed on a Weights and Biases run, which is created upon initialization. It 
is important to call the `wandb_finish` method before initializing a new 
`SeqClassTrainer` or `Evaluator` object, such that the previous W&B run can 
finish nicely before a new one is created. 

* `__init__(self, classification, reference, classifier=None, wandb_config={}, wandb_name=None)`: Initializes `Evaluator` instance for specified dataset (and model).
    * classification: The to-be-evaluated classification (prediction) (`pd.DataFrame`).
    * reference: Reference dataset with true labels (`pd.DataFrame` | `mycoai.Data`).
    * classifier: If provided and equipped with a .get_config method, will add its configuration information to the wandb run (Default is `None`).
    * wandb_config:  Extra information to be added to weights and biases config data (`dict`). 
    * wandb_name:  Name of the run to be displayed on weights and biases (`str`).
* `test(self, metrics=utils.EVAL_METRICS, levels=utils.LEVELS)`: Calculates classification performance in terms of specified metrics. Results are printed, 
returned as `pd.DataFrame`, and graphically displayed on W&B. 
* `detailed_report(self, level='species', train_data=None, latent_repr=None)`: Provides detailed information (on W&B) for one specified level. If `train_data`
is specified, will create graphs that plot compare the class-specific 
performance to its occurrence frequency in the dataset. If `latent_repr` is
specified, will create a 2D visualization of the inputted latent space with 
class-specific performance information that can be read when hovering over the
data points. The dimensions of the latent space are reduced by applying the 
t-distributed Stochastic Neighbourhood Embedding (t-SNE) algorithm to the first 
50 principal components of the latent space. 

### Visualizing a model's latent space
Visualizing the model's latent space creates a taxonomic map of the dataset that
can lead to insights in the model's inner workings and overall performance. Such
a visualization can be created as follows:

```python
import torch
from mycoai.data import Data
from mycoai.evaluate import Evaluator

test_data = Data('test.fasta') # Loading a test set
model = torch.load('model.pt') # Loads a pre-saved SeqClassNetwork object
classification = model.classify(test_data)
latent_repr = model.latent_space(test_data)

# Results will be visible on W&B
evaluator = Evaluator(classification, test_data, classifier=model)
evaluator.detailed_report('species', latent_repr=latent_repr)
```

## `mycoai.utils`
This module contains several package-wide constants and helper functions. Modifying some of the variables specified here can break the workings of the code. However, there are constants that can be set according to the user's preference:
* `VERBOSE`: controls the verbosity (how much is printed) of MycoAI. Set this value to 0 to hide most print statements (one of [0,1,2], default is 1).
* `PRED_BATCH_SIZE`: batch size for predictions (outside of training). Setting this value higher will lead to the consumption of more (GPU) memory, whereas setting this value lower will lead to slower performance (`int`, default is 64).
* `WANDB_PROJECT`: name of the W&B project to push the results to (`str`, default is 'ITS Classification'). 

Furthermore, the user might be interested in the following helper functions:
* `set_output_dir(path, parent='')`: Sets (global) output directory, creates new if it does not exist. Returns a string with the path to the newly created directory. Controls the `OUTPUT_DIR` variable, which avoids the accumulation of results files in your working directory. 
* `set_device(name)`: Sets (global) PyTorch device (either 'cpu', 'cuda', or 'cuda:0', 'cuda:1', etc.). Can be helpful to force the use of cpu even if you have a GPU available. By default and upon initialization, MycoAI will look for available GPUs and fall back to CPU only if no GPU is available. 
* `get_config(object=None, prefix='')`: Returns configuration information. Most MycoAI objects are equipped with a `get_config` method. This helper function calls the `get_config` method of the inputted `object`, returning a dictionary of configuration information such as hyperparameter settings. MycoAI uses this functionality to log and attach as much information as possible to Weights and Biases runs. However, this feature can also be very useful if you wish to document or know more about the settings of your model/training.

### Switching to CPU or a specific GPU
The example below demonstrates how you can use the `utils` module to force the use of CPU or a specific GPU:

```python
from mycoai import utils

utils.set_device('cpu') # Switches to cpu
utils.set_device('cuda:2') # Switches to the 3rd cuda device (3rd GPU)
```

# License
Distributed under the MIT License. See [LICENSE](/LICENSE) for more information.