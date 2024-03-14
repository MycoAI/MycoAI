'''Removes entries with specific accession numbers from UNITE data release
to serve as a held-out test set(s).

Example
-------
python -m scripts.paper.create_datasets \
       -u /data/UNITE_public_25.07.2023.fasta \
       -t /data/Supplementary_ITS_GBAccessionNumbers.txt \
          /data/CBSITS.current.classification.txt \
       -o /data/test1.fasta \
          /data/test2.fasta \
       -n /data/UNITE_public_25.07.2023_test_removed.fasta'''

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_datasets(unite_filepath, test_accession_files, test_output_files, 
                    trainset_filepath):

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
        yeast_data = pd.read_csv(test_accession_files[i], 
                                delimiter='\t')['GB accession number']
        merged = unite_data.merge(yeast_data, on='GB accession number', 
                                  how='inner')
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


if __name__== '__main__':

    parser = argparse.ArgumentParser(prog='python create_datasets.py',
        description=('Removes entries with specific accession numbers from' + 
                    ' UNITE data release to serve as a held-out test set(s).'))
    
    parser.add_argument('--unite_filepath', '-u',
        type=str,
        required=True,
        help='Path to full UNITE FASTA file.')
    
    parser.add_argument('--test_accession_files', '-t',
        type=str,
        nargs='+',
        required=True,
        help='Files containing accession numbers of to-be-removed sequences.')

    parser.add_argument('--test_output_files', '-o',
        type=str,
        nargs='+',
        required=True,
        help='Filepaths to where the newly produced test sets should be saved.')

    parser.add_argument('--trainset_filepath', '-n',
        type=str,
        required=True,
        help='Filepaths to where the reduced train set should be saved.')

    args = parser.parse_args()
    create_datasets(args.unite_filepath,
                    args.test_accession_files,
                    args.test_output_files,
                    args.trainset_filepath)