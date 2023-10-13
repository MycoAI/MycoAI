'''Classes for creating plots and visualizations.'''

import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import sklearn.metrics as skmetric
from . import utils, data

def counts_barchart(dataprep, level='phylum', id=''):
    '''Plots the number of sequences per class'''

    counts = dataprep.data.groupby(level, as_index=False)['id'].count().sort_values(
        'id', ascending=False)
    ax = counts.plot.bar(ylabel='# sequences', legend=False)
    ax.set_xticklabels(counts[level])
    plt.savefig(utils.OUTPUT_DIR + level + '_counts_' + id + '.png')
    plt.close()

def counts_boxplot(dataprep, id=''):
    '''Plots the number of sequences per class as boxplot (all taxon levels)'''

    fig, axs = plt.subplots(nrows=1,ncols=6,figsize=(9,3))

    for i in range(len(utils.LEVELS)):
        lvl = utils.LEVELS[i]
        counts = dataprep.data.groupby(lvl, as_index=False)[lvl].count()
        counts = counts.sort_values(lvl, ascending=False)
        counts.boxplot(ax=axs[i])

    axs[0].set_ylabel("# sequences")
    fig.suptitle('Taxon class counts')
    fig.tight_layout()
    plt.savefig(utils.OUTPUT_DIR + 'boxplot_' + id + '.png')
    plt.close()

def counts_sunburstplot(dataprep, id=''):
    '''Plots the taxonomic class distribution as a sunburst plot'''

    print("Creating sunburst plot...")
    counts = dataprep.data.groupby(utils.LEVELS, as_index=False).count()
    fig = px.sunburst(counts, path=utils.LEVELS, values='id')
    pio.write_image(fig, utils.OUTPUT_DIR + "sunburst_" + id + ".png", scale=4)

def classification_loss(history, target_levels):
    '''Plots the loss learning curves for a (multi-class) neural network'''
    for lvl in target_levels:
        valid_plot = plt.plot(history['Loss|valid|' + utils.LEVELS[lvl]],
                                        label=utils.LEVELS[lvl] + " (valid)")
        plt.plot(history['Loss|train|' + utils.LEVELS[lvl]], alpha=0.5,
                    color=valid_plot[0].get_color(), 
                    label=utils.LEVELS[lvl] + " (train)")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(utils.OUTPUT_DIR + '/loss.png')
    plt.close()
    return

def classification_metric(history, metric_name, target_levels):
    '''Plots the learning curves for a single metric on all target levels'''
    for level in [utils.LEVELS[i] for i in target_levels]:
        plt.plot(history[metric_name + '|valid|' + level], label=level)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(utils.OUTPUT_DIR + '/' + metric_name.lower() + '.png')
    plt.close()
    return

def confusion_matrices(model, data):
    '''Plots a confusion matrix for each predicted taxonomy level'''
    model.eval()
    with torch.no_grad():
        y_pred, y = model._predict(data, return_labels=True) 
        for i in range(len(y_pred)):
            argmax_y_pred = torch.argmax(y_pred[i].cpu(), dim=1)
            matrix = skmetric.confusion_matrix(y[:,i].cpu(), argmax_y_pred)
            plt.imshow(matrix)
            plt.savefig(utils.OUTPUT_DIR + '/' + 
                        utils.LEVELS[model.target_levels[i]] + '.png')
            plt.close()