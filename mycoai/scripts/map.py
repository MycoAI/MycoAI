'''Creates a 2D visualization of the final layer of the network, serving as 
a taxonomic map.'''

import torch
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mycoai.data import Data
from mycoai import utils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def map(fasta_filepath, output_filepath=utils.OUTPUT_DIR + 'map.html', 
        classification=None, context='data/trainset_valid.fasta', 
        model='models/MycoAI-multi-HLS.pt', device=utils.DEVICE):
    '''Maps the sequences in fasta_filepath on a 2D taxonomic space together
    with a set of reference sequences that serve as context.'''

    utils.set_device(device)

    # Loading the data and model
    model = torch.load('models/MycoAI-multi-HLS.pt', map_location=device)
    ref_data = Data(context, allow_duplicates=True)
    data = Data(fasta_filepath, tax_parser=None, allow_duplicates=True)
    if classification is not None:
        classification = pd.read_csv(classification)
        data.data[[f'{lvl}*' for lvl in utils.LEVELS]] = (
            classification[utils.LEVELS]
        )
        data.data['*'] = 'predicted'
    
    # Extracting the latent space and reducing to two dimensions
    ref_data_latent = model.latent_space(ref_data)
    data_latent = model.latent_space(data)
    latent = np.concatenate((ref_data_latent, data_latent), axis=0)
    latent = TSNE(2).fit_transform(PCA(50).fit_transform(latent))
    ref_data.data['Dim 1'] = latent[:len(ref_data.data),0].tolist()
    ref_data.data['Dim 2'] = latent[:len(ref_data.data),1].tolist()
    data.data['Dim 1'] = latent[len(ref_data.data):,0].tolist()
    data.data['Dim 2'] = latent[len(ref_data.data):,1].tolist()

    # Plotting
    ref_data = ref_data.data.drop(columns='sequence')
    data = data.data.drop(columns='sequence')
    ref_scatter = px.scatter(ref_data, x='Dim 1', y='Dim 2', 
                             hover_data=ref_data.columns, color='phylum')
    scatter = px.scatter(data, x='Dim 1', y='Dim 2', hover_data=data.columns, 
                         color_discrete_sequence=['black'])
    figure = go.Figure(data=(ref_scatter.data + scatter.data))
    figure.write_html(output_filepath, auto_play=False)
    print(f"Mapping saved to {output_filepath}.")


def main():

    parser = argparse.ArgumentParser(prog='python -m mycoai.scripts.map',
        description='Creates a 2D visualization of the final layer of the \
            network, serving as a taxonomic map.')
    
    parser.add_argument('fasta_filepath',
        help='Path to the FASTA file containing ITS sequences.')

    parser.add_argument('--out',
        default= [utils.OUTPUT_DIR + 'map.html'],
        type=str,
        nargs=1,
        help='Where to save the output to (default is map.html).')
    
    parser.add_argument('--classification',
        default= [None],
        type=str,
        nargs=1,
        help='Classification of input sequences in .csv (default is None).')
    
    parser.add_argument('--context',
        default= ['data/trainset_valid.fasta'],
        type=str,
        nargs=1,
        help='Set of sequences to serve as context in the taxonomic map. \
            (default is data/trainset_valid.fasta).')
    
    parser.add_argument('--model', 
        default=['models/MycoAI-multi-HLS.pt'],
        type=str,
        nargs=1,
        help="Path to saved SeqClassNetwork Pytorch model (default is \
              models/MycoAI-multi-HLS.pt).")
    
    parser.add_argument('--device', 
        default=[utils.DEVICE],
        type=str,
        nargs=1,
        help="Forces use of specific device (GPU/CPU). By default, MycoAI will\
              look for and use GPUs whenever available.")
    
    args = parser.parse_args()
    map(args.fasta_filepath, args.out[0], args.classification[0], 
        args.context[0], args.model[0], args.device[0])


if __name__ == '__main__':

    main()

