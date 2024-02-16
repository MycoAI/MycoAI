'''Classes for creating plots and visualizations.'''

import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import sklearn.metrics as skmetric
from mycoai import utils

def counts_barchart(dataprep, level='phylum', id=''):
    '''Plots the number of sequences per class'''

    data = dataprep.data[dataprep.data[level] != utils.UNKNOWN_STR]
    counts = data.groupby(level, as_index=False)['id'].count().sort_values('id', 
                                                                ascending=False)
    ax = counts.plot.bar(ylabel='# sequences', xlabel=level, width=1, 
                         color='#636EFA', figsize=(6,3), legend=False)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yscale('log')
    id = '_' + id if len(id) > 0 else ''
    plt.tight_layout()
    plt.savefig(utils.OUTPUT_DIR + level + '_counts' + id + '.pdf')
    plt.close()

def counts_boxplot(dataprep, id=''):
    '''Plots the number of sequences per class as boxplot (all taxon levels)'''

    fig, axs = plt.subplots(nrows=1,ncols=6,figsize=(9,3))
    id = '_' + id if len(id) > 0 else ''

    for i in range(len(utils.LEVELS)):
        lvl = utils.LEVELS[i]
        counts = dataprep.data.groupby(lvl, as_index=False)[lvl].count()
        counts = counts.sort_values(lvl, ascending=False)
        counts.boxplot(ax=axs[i])

    axs[0].set_ylabel("# sequences")
    fig.suptitle('Taxon class counts')
    fig.tight_layout()
    plt.savefig(utils.OUTPUT_DIR + 'boxplot' + id + '.png')
    plt.close()

def counts_sunburstplot(dataprep, id=''):
    '''Plots the taxonomic class distribution as a sunburst plot'''

    print("Creating sunburst plot...")
    counts = dataprep.data.groupby(utils.LEVELS, as_index=False).count()
    fig = px.sunburst(counts, path=utils.LEVELS, values='sequence')
    id = '_' + id if len(id) > 0 else ''
    fig.update_layout(width=500, height=500, margin = dict(t=0, l=0, r=0, b=0))
    pio.write_image(fig, utils.OUTPUT_DIR + "sunburst" + id + ".pdf", scale=1)

def classification_learning_curve(history, metric_name, levels, 
                                  show_valid=True, show_train=False):
    '''Plots the learning curves for a single metric on all specified levels'''
    
    for lvl in levels:
        valid_plot = False
        if show_valid:
            valid_plot = plt.plot(
                history[f'{metric_name}|valid|{utils.LEVELS[lvl]}'],
                label=utils.LEVELS[lvl] + " (valid)"
            )
        if show_train:
            if valid_plot is not False:
                plt.plot(history[f'{metric_name}|train|{utils.LEVELS[lvl]}'], 
                         alpha=0.5, color=valid_plot[0].get_color(), 
                         label=utils.LEVELS[lvl] + " (train)")
            else:
                plt.plot(history[f'{metric_name}|train|{utils.LEVELS[lvl]}'], 
                         alpha=0.5, label=utils.LEVELS[lvl] + " (train)")
    
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
                        utils.LEVELS[i] + '.png')
            plt.close()