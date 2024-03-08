'''Produces figures for the paper'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mycoai import utils
import plotly.express as px

plotly_color_scale = px.colors.qualitative.Plotly
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plotly_color_scale)

results_folder = '/data/luuk/results/'

# # Baseline comparison
# testsets = { # 'Validation': 'trainset_valid',
#             'Test set 1': 'test1',
#             'Test set 2': 'test2'}

# methods = {'MycoAI': 'MycoAI-multi LS 0.02 0.02 0.02 0.02 0.02 0.0',
#            'DNABarcoder': 'DNABarcoder',
#            'RDP classifier': 'RDP'} 

# metrics = ['Accuracy', 'Precision', 'Recall']

# results = pd.DataFrame()
# for testset_name in testsets:

#     test_file = testsets[testset_name]

#     for method_name in methods:

#         row = pd.read_csv(
#             results_folder + 
#             f'results/{test_file} {methods[method_name]}.csv'
#         )
#         row['testset_name'] = [testset_name]
#         row['method_name'] = [method_name]
#         results = pd.concat([results, row])
   
# figure, axs = plt.subplots(len(metrics), len(testsets), figsize=(12,6))
# width = 0.25

# for i, metric in enumerate(metrics):
    
#     for j, testset_name in enumerate(testsets):

#         axs[0,j].set_title(testset_name)
#         axs[i,j].set_axisbelow(True)
#         axs[i,j].grid(axis='y', which='major')

#         multiplier = 0
#         for method in methods:
#             data = results[(results['method_name'] == method) &
#                            (results['testset_name'] == testset_name)]
#             columns = [f'{metric}|test|{lvl}' for lvl in utils.LEVELS]
#             data = data[columns].iloc[0].values
#             offset = width*multiplier
#             axs[i,j].bar(
#                 x = np.arange(len(utils.LEVELS)) + offset,
#                 height = data,
#                 width = width,
#                 label = method
#             )
#             multiplier += 1
        
#         axs[i,0].set_ylabel(metric.capitalize())
#         if i < len(metrics)-1:
#             axs[i,j].set_xticks([])
#         else:
#             axs[i,j].set_xticks(np.arange(len(utils.LEVELS))+width,utils.LEVELS)
#         axs[i,j].set_ylim(0, 1.05)
#         axs[i,j].set_yticks(np.arange(1.05), minor=True)
 
# handles, labels = axs[0,0].get_legend_handles_labels()
# figure.legend(handles, labels, ncols=3, loc='upper left')
# figure.tight_layout(rect=(0,0,1,0.95)) 
# plt.savefig('baseline_comparison.png')
# plt.savefig('baseline_comparison.pdf')

# Latent spaces
import torch
from mycoai.data import Data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

utils.set_device('cuda:1')
model_folder = '/data/luuk/models/'

figure, axs = plt.subplots(2, 2, figsize=(12,10))

smooth = {'Standard Label Smoothing': 'LS 0.0 0.0 0.0 0.0 0.0 0.1.pt',
          'Hierarchical Label Smoothing': 'LS 0.02 0.02 0.02 0.02 0.02 0.0.pt'}

head = {'Single-head': 'MycoAI-single',
        'Multi-head': 'MycoAI-multi'}

for i, output in enumerate(head):

    for j, ls in enumerate(smooth):

        model_file = head[output] + ' ' + smooth[ls]
        model = torch.load(model_folder + model_file, utils.DEVICE)
        data = Data('/data/luuk/trainset_valid.fasta')
        latent_repr = model.latent_space(data)
        latent_repr = PCA(50).fit_transform(latent_repr)
        latent_repr = TSNE().fit_transform(latent_repr)
        data = data.data
        data['Dim 1'] = latent_repr[:,0]
        data['Dim 2'] = latent_repr[:,1]
    
        phyla = (data.groupby('phylum', as_index=False)
                .count().sort_values('id', ascending=False)
                .head(9)['phylum'].values)
        data.loc[~data["phylum"].isin(phyla), 'phylum'] = "Other"

        for phylum in data['phylum'].unique():
            if phylum == "Other" or phylum == "?": 
                continue
            axs[i,j].scatter('Dim 1','Dim 2',data=data[data['phylum']==phylum], 
                            s=3,label=phylum, rasterized=True)
        axs[i,j].scatter('Dim 1','Dim 2',data=data[data['phylum']=='Other'], 
                          s=3, label='Other', rasterized=True)
        axs[i,j].scatter('Dim 1','Dim 2',data=data[data['phylum']=='?'], 
                         color='black', s=3, label='?', rasterized=True)
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

        axs[0,j].set_title(ls)
        axs[i,0].set_ylabel(output, size='large')

handles, labels = axs[0,0].get_legend_handles_labels()
legend = figure.legend(handles, labels, loc='center right', markerscale=4.0,
                       title="Phylum")
legend.get_frame().set_linewidth(0.0)
legend._legend_box.align = "left"
figure.tight_layout(rect=(0,0,0.85,1))
plt.savefig(f'latent_spaces.png', dpi=300)
plt.savefig(f'latent_spaces.pdf', dpi=300)
    
# # Hierarchical label smoothing comparison
# models = {'Multi-head + HLS':  'MycoAI-multi LS 0.02 0.02 0.02 0.02 0.02 0.0',
#           'Multi-head':        'MycoAI-multi LS 0.0 0.0 0.0 0.0 0.0 0.1',
#           'Single-head + HLS': 'MycoAI-single LS 0.02 0.02 0.02 0.02 0.02 0.0', 
#           'Single-head':       'MycoAI-single LS 0.0 0.0 0.0 0.0 0.0 0.1'}

# width = 1/5
# multiplier = 0
# figure, ax = plt.subplots(1, figsize=(8,2.5))
# subtract = 0
# ax.grid(axis='y', which='major')
# ax.set_axisbelow(True)

# for i, model in enumerate(models):

#     for parents_inferred in [False, True]:

#         if model.startswith('Single-head') and not parents_inferred:
#             continue
#         modelname = models[model]
#         if model.startswith('Multi-head') and parents_inferred:
#             modelname += ' parents inferred'

#         data = pd.read_csv(results_folder + 'results/trainset_valid ' + 
#                            modelname + '.csv')
#         data = data[[f'Accuracy|test|{lvl}' for lvl in utils.LEVELS]]
#         data = data.iloc[0].values

#         offset = width*multiplier
#         if not parents_inferred:
#             ax.bar(np.arange(6) + offset, data, width, 
#                    color=plotly_color_scale[i], alpha=0.4)
#         else:
#             ax.bar(np.arange(6) + offset, data, width, 
#                    color=plotly_color_scale[i], label=model)

#     multiplier += 1

# plt.ylabel('Accuracy')
# # figure.tight_layout() 
# plt.xticks(np.arange(len(utils.LEVELS))+width,utils.LEVELS)
# handles, labels = ax.get_legend_handles_labels()
# figure.legend(handles, labels, ncols=4, loc='lower center')
# plt.tight_layout(rect=(0,0.1,1,1))
# plt.savefig('parents_inferred.png')
# plt.savefig('parents_inferred.pdf')

# # Big table
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']

# methods = {
#     'DNABarcoder': 'DNABarcoder',
#     'RDP classifier': 'RDP',
#     'MycoAI multi + HLS': 'MycoAI-multi LS 0.02 0.02 0.02 0.02 0.02 0.0',
#     'MycoAI multi': 'MycoAI-multi LS 0.0 0.0 0.0 0.0 0.0 0.1',
#     'MycoAI single + HLS': 'MycoAI-multi LS 0.02 0.02 0.02 0.02 0.02 0.0',
#     'MycoAI single': 'MycoAI-single LS 0.0 0.0 0.0 0.0 0.0 0.1'
# }

# testsets = {'Validation': 'trainset_valid',
#             'Test set 1': 'test1',
#             'Test set 2': 'test2'}

# table = []
# for metric in metrics:

#     for method in methods:

#         row = [metric, method]
#         methodfile = methods[method]

#         for testset in testsets:

#             testfile = testsets[testset]
#             data = pd.read_csv(results_folder + 'results/' + testfile + ' ' 
#                             + methodfile + '.csv')
#             data = data[[f'{metric}|test|{lvl}' for lvl in utils.LEVELS]]
#             row += list(data.values[0])

#         table.append(row)

# levels = ['P', 'C', 'O', 'F', 'G', 'S']
# table = pd.DataFrame(table, columns=['Metric', 'Method'] + levels*len(testsets))
# table.to_csv('table.csv', float_format="%.2f", index=False)