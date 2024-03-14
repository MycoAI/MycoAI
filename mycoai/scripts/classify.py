import argparse
import torch
from mycoai import utils

def classify(fasta_filepath, output_filepath=utils.OUTPUT_DIR+'prediction.csv', 
             model='models/MycoAI-multi-HLS.pt', device=utils.DEVICE):
    '''Predicts the taxonomies of sequences in file with specified method'''
    
    utils.set_device(device)
    deep_its_model = torch.load(model, map_location=utils.DEVICE)
    prediction = deep_its_model.classify(fasta_filepath)
    prediction.to_csv(output_filepath)
    print(f"Classification saved to {output_filepath}")
    

def main():

    parser = argparse.ArgumentParser(prog='python -m mycoai.scripts.classify',
        description='Taxonomic classification of fungal ITS sequences using a\
            deep neural network.')
    
    parser.add_argument('fasta_filepath',
        help='Path to the FASTA file containing ITS sequences.')

    parser.add_argument('--out',
        default= [utils.OUTPUT_DIR + 'prediction.csv'],
        type=str,
        nargs=1,
        help='Where to save the output to (default is prediction.csv).')
    
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
    classify(args.fasta_filepath, output_filepath=args.out[0], 
             model=args.model[0], device=args.device[0])


if __name__ == '__main__':

    main()