import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange 
import torch

def plot_attention_heatmap(args, tensor, interested_behavior, cmap='gray', extention=''): 
    tensor = rearrange(tensor, 'a b -> b a')
    # Set the plot size more suitably for a vertical display
    plt.figure(figsize=(40, 16))
    
    # Create a heatmap
    ax = sns.heatmap(tensor, cmap=cmap, cbar=True, annot=False)
    
    # Rotate x-axis labels to horizontal if needed, y-axis labels to vertical
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add title and labels if necessary
    plt.title('Attention Heatmap: '+args.subject_id+'  |  '+args.brain_region+'  |  '+interested_behavior, fontsize=30)
    plt.xlabel('Trails', fontsize=50)
    plt.ylabel('Neurons', fontsize=50)
    
    # Display the heatmap
    plt.savefig('../../outputs/neuropair.net'+extention+'/'+args.output_dir+'/attention_weights/attention_weights-'+interested_behavior, dpi=100)

def plot_attention_heatmap_mean(args, tensor, interested_behavior, cmap='gray', extention=''):
    tensor = rearrange(tensor, 'a b -> b a')
    # Set the plot size more suitably for a vertical display
    plt.figure(figsize=(20, 3))
    
    # Create a heatmap
    ax = sns.heatmap(tensor, cmap=cmap, cbar=True, annot=False)
    
    # Rotate x-axis labels to horizontal if needed, y-axis labels to vertical
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add title and labels if necessary
    plt.title('Attention Heatmap over Neurons on the x-Axis: '+args.subject_id+'  |  '+args.brain_region+'  |  '+interested_behavior, fontsize=20)
    # plt.xlabel('Neurons', fontsize=15)
    # ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xticks([])
    ax.set_yticks([]) 
    plt.savefig('../../outputs/neuropair.net'+extention+'/'+args.output_dir+'/attention_weights/attention_weights_mean-'+interested_behavior, dpi=100)