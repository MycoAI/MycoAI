''''Preprocesses the valid/test/reference data to be used by DNABarcoder and 
RDP Classifier.'''

import re
from mycoai.data import Data
from mycoai import utils

trainset = '/data/luuk/trainset.fasta'
testsets = ['/data/luuk/trainset_valid.fasta',
            '/data/luuk/test1.fasta',
            '/data/luuk/test2.fasta']

def create_identified_trainsets(path):
    '''Filters out unidentified sequences for each level, creating a BLAST 
    reference dataset with only identified taxonomic labels'''

    files = []
    data = Data(path, allow_duplicates=True)
    for level in utils.LEVELS:
        data = data.class_filter(level, remove_unidentified=True)
    filename = path.split('.')
    filename = f"{''.join(filename[:-1])}_identified.{filename[-1]}"
    data.export_fasta(filename)
    files.append(filename)

    return files

def testset_species(source, target):
    '''Creates a test set of samples identified at species-level.'''

    data = Data(source)
    data = data.class_filter('species', remove_unidentified=True)
    data.export_fasta(target)

    return target

def prep_reference(path):
    '''Preprocesses FASTA reference file into format for DNABarcoder'''

    file = open(path, 'r')
    source = file.readlines()
    file.close()
    target = open(path, 'w')

    i = 1
    for line in source:
        if line.startswith('>'):
            line = line.split('|')[1]
            line = re.sub('\?', 'unidentified', line)
            line = f'>{i} {line}\n'
            target.write(line)
            i += 1
        else:
            target.write(line)

def prep_testset(path):
    ''''Preprocesses FASTA test file into format for DNABarcoder/RDP'''

    file = open(path, 'r')
    source = file.readlines()
    file.close()
    path = path.split('.')
    target = open(path[0] + '_prepped.' + path[1], 'w')

    i = 1
    for line in source:
        if line.startswith('>'):
            target.write(f'>{i}\n')
            i += 1
        else:
            target.write(line)

for barcoder_trainset in create_identified_trainsets(trainset):
    prep_reference(barcoder_trainset)

testsets.append(testset_species(trainset, '/data/luuk/test3.fasta'))
            
for testset in testsets:
    prep_testset(testset)