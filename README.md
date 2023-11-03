
## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/MycoAI//data/escience/projects/dna-barcode/mycoai)](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-mycoai-00a3e3.svg)](https://www.research-software.nl/software/mycoai) [![workflow pypi badge](https://img.shields.io/pypi/v/mycoai.svg?colorB=blue)](https://pypi.python.org/project/mycoai/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=MycoAI_/data/escience/projects/dna-barcode/mycoai&metric=alert_status)](https://sonarcloud.io/dashboard?id=MycoAI_/data/escience/projects/dna-barcode/mycoai) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=MycoAI_/data/escience/projects/dna-barcode/mycoai&metric=coverage)](https://sonarcloud.io/dashboard?id=MycoAI_/data/escience/projects/dna-barcode/mycoai) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects//data/escience/projects/dna-barcode/mycoai/badge/?version=latest)](https:///data/escience/projects/dna-barcode/mycoai.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/build.yml/badge.svg)](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/cffconvert.yml/badge.svg)](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/sonarcloud.yml/badge.svg)](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/markdown-link-check.yml/badge.svg)](git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai/actions/workflows/markdown-link-check.yml) |

## How to use mycoai

<<<<<<< HEAD
Using AI and machine learning to analyze patterns in biodiversity genomics in Fungi
=======
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
>>>>>>> master

The project setup is documented in [project_setup.md](project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

## Installation

To install mycoai from GitHub repository, do:

<<<<<<< HEAD
```console
git clone git@github.com:MycoAI//data/escience/projects/dna-barcode/mycoai.git
cd /data/escience/projects/dna-barcode/mycoai
python -m pip install .
=======
The main script, [`classify.py`](/scripts/classify.py) can be used for the
assignment of taxonomic labels to ITS sequences within a FASTA file. The output
will be saved in a 'prediction.csv' file. 
The script takes the following arguments:

| Argument | Required | Description | Values | 
| ---  | --- | --- | --- |
| `fasta_filepath` | Yes | Path to the FASTA file containing ITS sequences. | path 
| `--out`   | No | Where to save the output to.| path
| `--method` | No | Which classification method to use (default is 'deep_its'). | ['deep_its']

## Using the package
Users can load a FASTA file into a `mycoai.data.DataPrep` object, which 
comes with several data filtering methods and can encode the data into a format 
that is suitable for the desired classifier. By default, it is assumed that the 
FASTA headers contain labels following the [UNITE](https://unite.ut.ee/) format,
but the `DataPrep` object also allows for 1) unlabelled FASTA sequence files or 
2) custom header parsers functions written by the user. 

### Deep ITS classifiers
The `mycoai.models.ITSClassifier` class uses deep neural networks for its predictions. 
It can be configured in multiple ways, its arguments are listed below. The most
important elements of a Deep ITS classifier are its data encoding methods, and 
its base architecture.  

| Argument | Description | Values | 
| --- | --- | --- |
| `base_arch` | The main neural network | `torch.nn.Module` instance | 
| `dna_encoder` | The applied DNA encoding method | One of ['4d', 'bpe', 'kmer-tokens', 'kmer-onehot', 'kmer-spectral'] or a `DNAEncoder` instance |
| `tax_encoder` | The label encoder used for the (predicted) labels | 'categorical' or a `TaxonEncoder` instance |
| `fcn_layers` | List of node numbers for fully connected neural network before the output head | `list[int]` of any length | 
| `output` | The type of output head(s) for the neural network | One of ['single', 'multi', 'chained', 'inference']
| `target_levels` | Names of the taxon levels for the prediction tasks | `list[str]` with one or more of ['phylum', 'class', 'order', 'family', 'genus', 'species']
| `dropout` | Dropout percentage for the dropout layer | `float` in [0,1]

#### Encoding methods
A neural network operates on numbers, which is why the input data must be
converted from an alphabetical sequence (mostly consisting of [A,C,T,G]) into a
numerical sequence. The same applies to the taxonomic classes: internally, the 
network refers to them as numbers. For reusability, the applied encoding method 
is always contained within the model. 

For encoding DNA sequences, several alternative algorithms are included within 
MycoAI:
* 4D: The `FourDimDNA` class converts DNA sequences of length $l$ into 
$(4×l)$-dimensional vectors, yielding a 4-channel representation (comparable to
the 3-channel RGB representation of images).
* $k$-mer based: initializes a dictionary of all possible sequences of length 
$k$, then converts the sequence into:
    * Tokens: where the occurrence of a specific $k$-mer is indicated by a 
    unique index (`KmerTokenizer`).
    * One-hot encoding: yielding a sequence of sparse vectors with a 1 at the
    corresponding $k$-mer index (`KmerOneHot`).
    * Spectrum: with the occurrence frequency per $k$-mer (`KmerSpectrum`).
* Byte Pair Encoding (BPE): The `BytePairEncoder` keeps track of the most 
frequently appearing combinations of characters, resulting in a fixed-sized 
vocabulary of flexibly-sized words. The occurrence of a word is indicated by a 
unique index (token).   

#### Base architectures
The base architectures can be described below. Their hyperparameters (e.g. 
kernel sizes) can be configured individually.

| Name | Description | Supported encoding methods |
| --- | --- | --- |
`SimpleCNN` | A simple convolutional neural network with batch normalization and max-pooling layers.  | `FourDimDNA`, `KmerOneHot`, `KmerSpectrum`
`ResNet` | A CNN with residual connections between layers. | `FourDimDNA`, `KmerOneHot`, `KmerSpectrum`
`BERT` | A transformer-based encoder, applying attention mechanisms.  | `KmerTokenizer`, `BytePairEncoder`

#### Training a Deep ITS classifier
A deep ITS classifier can be trained on labelled data by using the 
`ClassificationTask.train` method. Custom/weighted data sampler or loss
functions can be specified. For example, by using `Dataset.weighted_loss`, the 
loss for each taxonomic class is weighted by the reciprocal class size 
(accounting for class imbalance). The `ClassificationTask.train` method will 
return both the trained model and a history dataframe, containing values for 
several metrics collected during the training epochs. These can be plotted using 
the functions available in `plotter`. For an example, see below. 

#### Example
For a more extensive example, covering more options, we refer to 
[example.py](/example.py).

```python
import torch
from mycoai import data, plotter
from mycoai.models import ITSClassifier
from mycoai.models.architectures import ResNet
from mycoai.training import ClassificationTask

# Data import & preprocessing
train_data = data.DataPrep('/data/s2592800/test1.fasta')
train_data, valid_data = train_data.encode_dataset('4d', valid_split=0.2)

# Use encoding scheme from train_data on the test set
test_data = data.DataPrep('/data/s2592800/test2.fasta')
test_data = test_data.encode_dataset(dna_encoder=train_data.dna_encoder,
                                     tax_encoder=train_data.tax_encoder)

# Model definition
arch = ResNet([2,2,2,2]) # = ResNet18
# This model will have a single output head and make genus-level predictions
model = ITSClassifier(arch, train_data.dna_encoder, train_data.tax_encoder,  
               target_levels=['genus'], fcn_layers=[128,20,64], output='single')

# Train/test (optionally with weighted loss/sampling) 
model, history = ClassificationTask.train(model, train_data, valid_data, 100)
plotter.classification_loss(history, model.target_levels)
result = ClassificationTask.test(model, test_data)
>>>>>>> master
```

## Documentation

Include a link to your project's full documentation here.

## Contributing

If you want to contribute to the development of mycoai,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
=======
# MycoAI: ITS classification with deep learning techniques
Python package for classifying fungal ITS metabarcoding sequences into their 
corresponding taxonomies, using deep neural networks.
