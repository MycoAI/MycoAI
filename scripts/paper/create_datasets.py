'''Removes entries with specific accession numbers from UNITE data release
to serve as a held-out test set(s).'''

import pandas as pd
import numpy as np
from tqdm import tqdm

unite_filepath = '/data/s2592800/UNITE_public_29.11.2022.fasta'
test_accession_files = ['/data/s2592800/Supplementary_ITS_GBAccessionNumbers.txt', 
                        '/data/s2592800/CBSITS.current.classification.txt']
test_output_files = ['/data/s2592800/test1.fasta', '/data/s2592800/test2.fasta']
trainset_filepath = '/data/s2592800/UNITE_public_29.11.2022_test_removed.fasta'

# Turn original UNITE file into pd dataframe
unite_file = open(unite_filepath).readlines()
unite_data = []
for i in tqdm(range(0,len(unite_file),2)):
    accession = unite_file[i].split('|')[0][1:]
    unite_data.append([accession,int(i)])
unite_data = pd.DataFrame(unite_data, columns=['GB accession number', 'i'])

all_indices = []
for i in range(len(test_accession_files)):

    # Read test data and perform inner join, extract indices
    yeast_data = pd.read_csv(test_accession_files[i], delimiter='\t')['GB accession number']
    merged = unite_data.merge(yeast_data, on='GB accession number', how='inner')
    indices = merged['i'].to_numpy()
    indices = np.concatenate([indices,indices+1])
    indices = list(np.sort(indices))

    # Write test file
    test_file_contents = [unite_file[j] for j in indices]
    test_file = open(test_output_files[i], 'w')
    test_file.writelines(test_file_contents)
    test_file.close()

    all_indices += indices

# Write new UNITE data file with test data removed
train_file = open(trainset_filepath, 'w')
iterator = 0
all_indices = list(np.sort(indices))
all_indices.append(-1)
for i in tqdm(range(0, len(unite_file))):
    if i == all_indices[iterator]:
        iterator += 1
    else:
        train_file.write(unite_file[i])
test_file.close()