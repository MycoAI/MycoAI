*Note: this repository is under active development. Parts of this description
may not be up-to-date with the latest changes in the code. [Last update:
8-11-2023]*

# About MycoAI
Python package for classification of fungal ITS metabarcoding sequences. MycoAI 
introduces a collection of deep neural classifiers and allows users to train
their own. 
<!-- FUTURE Traditional methods, such as BLAST (+ DNABarcoder) and RDPClassifier
are also included. The package serves as a standardized comparison platform 
that supports the user in picking the best-suitable classifier for the task at 
hand. -->

# Installation
Currently, the only way of using MycoAI is from source:
    
    git clone https://github.com/MycoAI/MycoAI

You can install the specified requirements manually, or create a conda 
environment with all the necessary dependencies using the command below. 

    conda env create -f environment.yml

## Requirements
* Python version 3.8 or higher [[link](https://www.python.org/)]
* Pandas [[link](https://pandas.pydata.org/)]
* Plotly [[link](https://plotly.com/python/)]
* PyTorch [[link](https://pytorch.org/)]
* Matplotlib [[link](https://matplotlib.org/)]
* Numpy [[link](https://numpy.org/)]
* Scikit-learn [[link](https://scikit-learn.org/)]
* Scipy [[link](https://scipy.org/)]
* SentencePiece [[link](https://github.com/google/sentencepiece)]
* Tqdm [[link](https://github.com/tqdm/tqdm)]
* Weights and Biases [[link](https://wandb.ai/site)]

# Usage
The MycoAI package was designed for two main usage scenarios:
1. [**Running pre-written scripts**](##running-pre-written-scripts): In case you 
goal (e.g. dataset classification or model training), you can use one of the 
scripts provided within the [scripts](/scripts) folder.  
2. [**Writing own scripts**](##writing-own-scripts): In case you want to use the
(many) available options within the MycoAI package (or expand upon them), you 
can use the package to write your own scripts or modify existing scripts to your
demands. 

Whatever approach you follow, you can use MycoAI for the classification of your 
ITS datasets, training different types of models, and/or comparing their 
performances. 

## Running pre-written scripts
You can use pre-written scripts to train and classify ITS sequences using AI and non-AI (dnabarcoder) methods
Scripts are available within the [scripts](./scripts) folder. Scripts can be run 
through the following prompt:

    python -m its_classifier <subcommand> <subcommand args>

Two subcommands are available: 
* `train_deep`: trains a deep neural network for ITS classification.
* `classify_deep`: classifies ITS sequences using a trained model.

### Training
To train a deep neural network for ITS classification, run:

    python -m  its_classifier train_deep <subcommand args> <path to the FASTA file containing ITS sequences for training>
The arguments for `train_deep` subcommand are as follows:

| Argument             | Required | Description                                                                                                                                                                                                                                                                      | Values                                                       | 
|----------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `--save_model`       | No       | Path to where the trained model should be saved. (default is model.pt)                                                                                                                                                                                                           | `path`                                                       | 
| `--base_arch_type`   | No       | Type of the to-be-trained base architecture (default is BERT).                                                                                                                                                                                                                   | ['ResNet', 'BERT', 'CNN']                                    |
| `--validation_split` | No       | Fraction of data to be used for validation (default is 0.2).                                                                                                                                                                                                                     | `float`                                                      | 
| `--target_levels`    | No       | A list of yaxonomic levels to predict. (default is all levels).                                                                                                                                                                                                                  | `['phylum', 'class', 'order', 'family', 'genus', 'species']` | 
| `--metrics`          | No       | Evaluation metrics to report during training, provided as dictionary with metric name as key and function as value, forexample {'Accuracy': skmetric.accuracy_score, 'MCC': skmetric.matthews_corrcoef} (default is accuracy, balanced acuracy, precision, recall, f1, and mcc). | `dict`                                                       | 
| `--wandb_config`     | No       | Allows the user to add extra information to the weights and biases config data (default is {}).                                                                                                                                                                                  | [0,1]                                                        |
| `--wandb_name`       | No       | Name of the run to be displayed on weights and biases. Will choose a random name if unspecified (default is None).                                                                                                                                                               | [0,1]                                                        |
| `--gpu`              | No       | Use CUDA enabled GPU if available (default is None). The number following the argument indicates the GPU to use in a multi-GPU system.                                                                                                                                           | `int`                                                        |

The values for the hyperparameters are in [hyperparameters.ini](./scripts/hyperparameters.ini). Edit the file to change the default values.

### Classification
To classify ITS sequences using a trained model, run:

    python -m  its_classifier classify_deep <subcommand args> <path to theFASTA file containing ITS sequences for classification>
The arguments for `classify_deep` subcommand are as follows:

| Argument       | Required | Description                                                                                                                            | Values | 
|----------------|----------|----------------------------------------------------------------------------------------------------------------------------------------|--------|
| `--load_model` | Yes      | Path to model to load.                                                                                                                 | `path` |
| `--out`        | No       | Path to the output CSV file to save the classification results (default is predictions.csv).                                           | `path` |
| `--gpu`        | No       | Use CUDA enabled GPU if available (default is None). The number following the argument indicates the GPU to use in a multi-GPU system. | `int`  | 

### Evaluation script
To evaluate the quality of a classification (provided as .csv file) predicted by
a (deep/non-deep) ITS classifier, run:

    python -m evaluate <Path to .csv file containing predicted labels> <Path to .csv or FASTA file containing ground truth labels>


## Writing own scripts
The MycoAI package can be used to write your own scripts for training and classifying ITS sequences using depp AI models.
### Overview
To train/evaluate a model on a dataset, these are the steps that you can follow
when writing your own scripts.
1. [Importing the data](#importing-the-data)
2. [Applying data filters](#applying-data-filters)
3. [Data conversion into format used by classifier](#data-encoding)
4. Configuring/training your ITS classifier
    - [Deep-learning-based](#deep-learning-based-its-classifiers) classifier
        1. [Configuration](#model-configuration)
        2. [Pre-training](#pre-training)
        3. [Training](#training)
    - [Traditional](#traditional-its-classifiers) classifier
5. [Performance evaluation](#performance-evaluation)

### Importing the data
Users can load a FASTA file into a `mycoai.data.Data` object, which 
comes with several data filtering methods and can encode the data into a format 
that is suitable for the desired classifier. By default, it is assumed that the 
FASTA headers contain labels following the [UNITE](https://unite.ut.ee/) format,
but the `DataPrep` object also allows for: 
1) unlabelled FASTA sequence files or 
2) custom header parsers functions written by the user.
but the `Data` object also allows for custom header parsers functions written by

##### Example
Assume we have two files: 
- `dataset1.fasta`: following the same taxonomic label notation as used in 
UNITE: \
">KY106084|k__Fungi;p__Ascomycota;c__Saccharomycetes;o__Saccharomycetales;f__Saccharomycetaceae;g__Zygotorulaspora;s__Zygotorulaspora_florentina|SH0987707.09FU"
- `dataset2.fasta`:  using a comma-separated notation for the header: \
">SH0987707.09FU,Ascomycota,Saccharomycetes,Saccharomycetales,Saccharomycetaceae,Zygotorulaspora,Zygotorulaspora_florentina"

The first dataset can be loaded out-of-the-box. For the second dataset, we need 
to define a custom parsing function:

```python
import mycoai

# For data with labels following the UNITE format
unite_data = mycoai.data.Data('dataset1.fasta')

# For data with labels following a different format
def custom_parser(fasta_header):
    '''Example parsing function for header with comma-separated labels'''
    return fasta_header[1:].split(",")

own_data = mycoai.data.Data('dataset2.fasta', tax_parser=custom_parser)
```

Note that the `Data` object assumes that `tax_parser` returns a list of the 
following format: [id, phylum, class, order, family, genus, species].

### Applying data filters
The `Data` class contains the following filter methods:
1. `class_filter`: Used to manipulate the size of taxonomic classes, designed
for creating a smaller-sized data subset. It will retains at most `max_samples`
sequences at the specified taxon level, from classes with at least `min_samples` 
available. It can also randomly select a `max_classes` number of classes.
2. `sequence_quality_filter`: Removes sequences with more than a tolerated ratio
(`tolerance`) of uncertain bases (bases that are not in [A,C,G,T]).
3. `sequence_length_filter`: Removes sequences with more than the tolerated 
standard deviation (`tolerance`) from the mean length *or* outside of the 
specified range (in this case `tolerance` must be of type `list` or `RangeType`)

Note for users that wish to implement their own filtering methods: a `Data`
object has a `data` attribute which is a pandas Dataframe. Filtered datasets can
be saved as fasta files in UNITE format using the `.export_fasta` method.

#### Example
```python
import mycoai

data = mycoai.Data('dataset1.fasta') # Load data

# Select a subset of 1000 species from those that have at least 5 samples
data = data.class_filter('species', min_samples=5, max_classes=1000)
# Remove sequences with more than 5% of bases not in [A,C,G,T]
data = data.sequence_quality_filter(tolerance=0.05)
# Remove sequences with more than 4 stds from the mean length
data = data.sequence_length_filter(tolerance=4)
```

### Data encoding
Each classifier requires its own input format. A `mycoai.Data` object has
the following methods for converting its data into the right encoding:
* `encode_dataset`: for deep-learning-based classifiers.
* More will be added soon.

#### Data encoding for deep neural classifiers
A neural network operates on numbers, which is why the input data must be
converted from an alphabetical sequence (mostly consisting of [A,C,T,G]) into a
numerical sequence. The same applies to the taxonomic classes: internally, the 
network refers to them as numbers. The model must contain the applied encoding 
method, such that it can encode new DNA input and decode its predictions into a 
human-interpretable format whenever new data comes in. 

The `encode_dataset` method from `mycoai.data.Data` returns a 
`mycoai.data.TensorData` object which can be inputted to the neural network (as 
it inherits from `torch.utils.data.Dataset`). The most important argument of the 
`encode_dataset` method is `dna_encoder`. The DNA encoding methods implemented 
within `mycoai.data.encoders` are listed [below](#dna-encoding-methods). 
`encode_dataset` also has a `valid_split` argument that allows users to define 
assign a random fraction of the data for validation. Furthermore, it has an 
`export_path` argument which can be used to save encoded datasets. These 
datasets can then later be loaded using the `filepath` argument of the 
`mycoai.data.TensorData` constructor. 

#### DNA encoding methods
| Name              | Description                                                                                                                                                                                              | Tensor shape* | Example encoding: ACGACGT                                                   |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|-----------------------------------------------------------------------------|
| `FourDimDNA`      | 4-channel representation, comparable to the 3-channel RGB representation of images.                                                                                                                      | $[n,4,l]$     | `[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]` |
| `BytePairEncoder` | Keeps track of the most frequently appearing combinations of characters, resulting in a fixed-sized vocabulary of flexibly-sized words. The occurrence of a word is indicated by a unique index (token). | $[n,l]$       | `[[1,5,6,2]]` assuming tokens 'ACGA' and 'CGT'                              | 
| `KmerTokenizer`   | Encodes each possible $k$-mer with a unique index (token)                                                                                                                                                | $[n,l]$       | `[[1,5,5,2]]` for $k=3$                                                     |
| `KmerOneHot`      | One-hot encoding of $k$-mers, given each possible $k$-mer its own channel                                                                                                                                | $[n,4^k,l]$   | `[[[0,...,1,...,0],[0,...,1,...,0]]]`                                       |
| `KmerSpectrum`    | Encodes each sequence into a frequency vector for its $k$-mers.                                                                                                                                          | $[n,1,4^k]$   | `[[[0,...,2,...]]]`                                                         |

**=for $n$ sequences with (padded) length $l$*

Note that in case of `BytePairEncoder` and `KmerTokenizer`, five token values
have special meanings. For example, in the table above 1 indicates the CLS 
(start) token and 2 indicates the SEP (end) token. We also use dedicated tokens 
for padding, unknown values, and masking.  

#### Example
```python
import mycoai

dataprep = mycoai.data.Data('dataset1.fasta') # Load data

# Using BytePair encoding with default settings
dataset = dataprep.encode_dataset('bpe')

# Modifying the parameters of BytePair encoding
dna_encoder = mycoai.encoders.BytePairEncoder(dataprep, vocab_size=1000)
example = dna_encoder.encode('ACGACGT')
dataset = dataprep.encode_dataset(dna_encoder)

# Exporting/loading train/validation data
train_data, val_data = dataprep.encode_dataset('bpe', valid_split=0.1, 
                                               export_path=['tr.pt', 'val.pt'])
train_data = mycoai.data.TensorData('tr.pt')
val_data = mycoai.data.TensorData('val.pt')
```

### Deep-learning-based ITS classifiers
The `mycoai.deep.models.DeepITSClassifier` class uses deep neural networks for 
its predictions. MycoAI offers various options for the user to 
[configure](#model-configuration) and [train](#training) his/her own neural ITS 
classifier. Any `torch.nn.Module` object can be used as a basis for such a 
classifier. The package also includes [pre-training](#pre-training) options for 
BERT-like architectures.  

#### Model configuration
The `mycoai.deep.models.DeepITSClassifier` class can be configured in multiple 
ways, its arguments are listed below. The most important elements of a Deep ITS 
classifier are its data [encoding](#data-encoding-for-deep-neural-classifiers) 
methods, and its base architecture.  

| Argument        | Description                                                                    | Values                                                                                         | 
|-----------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `base_arch`     | The main neural network                                                        | `torch.nn.Module` instance                                                                     | 
| `dna_encoder`   | The applied DNA encoding method                                                | One of ['4d', 'bpe', 'kmer-tokens', 'kmer-onehot', 'kmer-spectral'] or a `DNAEncoder` instance |
| `tax_encoder`   | The label encoder used for the (predicted) labels                              | 'categorical' or a `TaxonEncoder` instance                                                     |
| `fcn_layers`    | List of node numbers for fully connected neural network before the output head | `list[int]` of any length                                                                      | 
| `output`        | The type of output head(s) for the neural network                              | One of ['single', 'multi', 'chained', 'inference']                                             |
| `target_levels` | Names of the taxon levels for the prediction tasks                             | `list[str]` with one or more of ['phylum', 'class', 'order', 'family', 'genus', 'species']     |
| `dropout`       | Dropout percentage for the dropout layer                                       | `float` in [0,1]                                                                               |


Any `torch.nn.Module` object can be used as a base architecture, which allows
the user to configure his/her own model type as an ITS classifier. The package
comes with a number of pre-implemented base architectures, which are described 
below. Their hyperparameters (e.g. kernel sizes) can be configured individually.

| Name        | Description                                                                            | Supported encoding methods                 |
|-------------|----------------------------------------------------------------------------------------|--------------------------------------------|
| `SimpleCNN` | A simple convolutional neural network with batch normalization and max-pooling layers. | `FourDimDNA`, `KmerOneHot`, `KmerSpectrum` |
| `ResNet`    | A CNN with residual connections between layers.                                        | `FourDimDNA`, `KmerOneHot`, `KmerSpectrum` |
| `BERT`      | A transformer-based encoder, applying attention mechanisms.                            | `KmerTokenizer`, `BytePairEncoder`         |

Depending on the nature of the task, users might want to predict multiple
taxonomic levels using the same neural network. To this end, we implemented
several types of output heads in `mycoai.deep.models.output_heads`. The output 
head, or a string indicating the type, is inputted to `DeepITSClassifier`'s 
constructor.
* `SingleHead` / 'single': Standard classification output head: a 
softmax-activated layer in which the number of nodes equals the number of 
classes to predict.
* `MultiHead` / 'multi': Six independent, standard output heads, one per 
taxonomic target level. 
* `ChainedMultiHead` / 'chained': Like 'multi', except that the output head
corresponding to a taxonomic level is not only inputted with the output from the 
base architecture, but *also* with the output from the head corresponding to its 
parent taxon level. For example, the class-level output head gets phylum-level 
predictions as extra input, i.e. they are chained. 
* `SumInference` 'infer_sum': A single standard classification output head, 
but the higher taxon levels are inferred from the data. The inference is done
by multiplying the output with inference matrices that describe how often in the 
training data a certain lower-level taxon belonged to a certain higher-level 
taxon. These inference matrices are part of the `TaxonEncoder` class and 
calculated during data encoding. 
* `ParentInference` 'infer_parent': Infers parent classes by looking in the 
inference matrix and seeing what parent a child class is most often part of.

#### Example
```python
#TODO
```

### Pre-training
TODO

#### Example
```python
#TODO
```

### Training
A deep ITS classifier can be trained on labelled data by using the 
`DeepITSTrainer.train` method. Please find its input arguments below.

| Argument | Description | Values | 
| --- | --- | --- |
| `model` | Neural network | `mycoai.deep.models.DeepITSClassifier` | 
| `train_data` | Dataset containing encoded ITS sequences for training | `mycoai.data.TensorData` | 
| `valid_data` | If provided, uses this dataset containing encoded ITS sequences for validation | `mycoai.data.TensorData`, default is `None` | 
| `epochs` | Number of training iterations | `int`, default is 100 | 
| `loss` | To-be-optimized loss function (or list of functions per level) | Callable or list of callables per level, default is `CrossEntropyLoss` | 
| `batch_size` | Number of training examples per optimization step | `int`, default is 64 |
| `sampler` | Strategy to use for drawing data samples | `torch.utils.data.Sampler`, default is random | 
| `optimizer` | Optimization strategy | `torch.optim`, default is Adam | 
| `metrics ` | Evaluation metrics to report during training, provided as dictionary with metric name as key and function as value | `dict{str:callable}`, default is accuracy, balanced acuracy, precision, recall, f1, and mcc. | 
| `weight_schedule` | Factors by which each level should be weighted in loss per epoch | `mycoai.deep.train.weight_schedules`, default is `Constant([1,1,1,1,1,1])` |  
| `warmup_steps` | When specified, the lr increases linearly for the first `warmup_steps` then decreases proportionally to $1/\sqrt{step}$. Works only for models with `d_model` attribute (e.g. BERT) | `int`, default is `None` |  
| `wandb_config` | Allows the user to add extra information to the weights and biases config data. | `dict{str:str}`, default is `{}` | 
| `wandb_name` | Name of the run to be displayed on weights and biases. Will choose a random name if unspecified. | `str`, default is `None` |

<!-- 
TODO: explain some of the arguments in more detail

WEIGHTS AND BIASES

SAMPLER

WEIGHTED LOSS

A deep ITS classifier can be trained on labelled data by using the 
`DeepITSTrainer.train` method. Custom/weighted data sampler or loss
functions can be specified. For example, by using `TensorData.weighted_loss`, the 
loss for each taxonomic class is weighted by the reciprocal class size 
(accounting for class imbalance).  -->

The `DeepITSTrainer.train` method will return both the trained model and a 
history dataframe, containing values for several metrics collected during the 
training epochs. These can be plotted using the functions available in 
`plotter`. For an example, see below. 

### Example
For a more extensive example, covering more options, we refer to 
[example.py](/example.py).

```python
import torch
from mycoai import data, plotter
from mycoai.deep.models import DeepITSClassifier
from mycoai.deep.models import ResNet
from mycoai.deep.train import DeepITSTrainer

# Data import & preprocessing
train_data = data.Data('/data/s2592800/test1.fasta')
train_data, valid_data = train_data.encode_dataset('4d', valid_split=0.2)

# Use encoding scheme from train_data on the test set
test_data = data.Data('/data/s2592800/test2.fasta')
test_data = test_data.encode_dataset(dna_encoder=train_data.dna_encoder,
                                     tax_encoder=train_data.tax_encoder)

# Model definition
arch = ResNet([2,2,2,2]) # = ResNet18
# This model will have a single output head and make genus-level predictions
model = DeepITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder,  
               target_levels=['genus'], fcn_layers=[128,20,64], output='single')

# Train/test (optionally with weighted loss/sampling) 
model, history = DeepITSTrainer.train(model, train_data, valid_data, 100)
plotter.classification_loss(history, model.target_levels)
result = DeepITSTrainer.test(model, test_data)
```

### Traditional ITS classifiers
Soon, alternative methods like BLAST (+DNABarcoder) and RDP classifier will be included in MycoAI. 

#### Example
```python
#TODO
```

## Performance evaluation

### Example
```python
#TODO
```
## Technical aspects

## Utils

## Example

<!-- ### Deep learning details? -->

<!-- FUTURE # Contributing -->

## License
Distributed under the MIT License. See [LICENSE](/LICENSE) for more information.

## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).

## Contact
Luuk Romeijn [[e-mail](mailto:l.romeijn@umail.leidenuniv.nl)]
Nauman Ahmed [[e-mail](mailto:n.ahmed@esciencecenter.nl)]