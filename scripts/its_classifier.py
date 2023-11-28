import train
import classify
import argparse




if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='python its_classifier.py',
        description='Taxonomic classification of fungal ITS sequences.', usage="%(prog)s <positional arguments> [options]")

    '''Predicts the taxonomies of sequences in file with specified method'''

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for the "blast" command
    train_blast_parser = subparsers.add_parser('train_blast')

    # Subparser for the "deep_its" command
    train_deep_its_parser = subparsers.add_parser('train_deep')

    # Subparser for the "its" command
    classify_blast_parser = subparsers.add_parser('classify_blast')


    # Subparser for the "its" command
    classify_deep_its_parser = subparsers.add_parser('classify_deep')


    trainer = train.Train(train_blast_parser, train_deep_its_parser)
    trainer.add_train_blast_args()
    trainer.add_train_deep_args()
    #classify.add_classify_args(classify_parser)

    args = parser.parse_args()

    if args.command == 'train_deep':
        trainer.train_deep(args)
    #elif args.command == 'classify':
     #   classify.classify(args)