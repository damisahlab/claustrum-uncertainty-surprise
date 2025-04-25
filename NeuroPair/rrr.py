# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:43:23 2024

@author: rd883
"""
#%% Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import glob
import pandas as pd
from einops import rearrange
from scipy.io import loadmat
from scipy.stats import bootstrap
from scipy.spatial.distance import pdist, squareform 
from IPython import get_ipython
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

#%% Functions
class AttentionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_neurons):
        super(AttentionHead, self).__init__() 
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)  
        self.neuron_behaviour_alpha = nn.Parameter(torch.rand(num_neurons, output_size))  
    
    def forward(self, x): 
        
        x = F.relu(self.layer1(x)) 
        x = self.layer2(x)
 
        neuron_behaviour_alpha = self.neuron_behaviour_alpha.repeat(x.shape[0], 1, 1)
        neuron_behaviour_alpha = neuron_behaviour_alpha.softmax(-1) 
        out = neuron_behaviour_alpha*x 

        return out, neuron_behaviour_alpha
 
class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features, out_features, n_heads):
        """
        Implementaion of a multi-head graph attention mechanism. 
        
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node
            n_heads (int): Number of attention heads to use for parallel attention processes
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_hidden = out_features
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, zh, adj_mat):
        """
        Propagates the input through the graph attention layer to compute the output features and attention scores.

        Args:
            zh (torch.tensor): Node features tensor,  
            adj_mat (torch.tensor): Adjacency matrix of the graph

        Returns:
            torch.tensor: Aggregated node features after applying attention 
            torch.tensor: Attention scores for each node pair
        """
        
        g_l = self.linear_l(zh).view(zh.shape[0], self.n_heads, self.n_hidden)
        g_r = self.linear_r(zh).view(zh.shape[0], self.n_heads, self.n_hidden)

        # Create repeated arrays
        g_l_repeat = g_l.repeat(zh.shape[0], 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(zh.shape[0], dim=0)

        # Sum features from transformations
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(zh.shape[0], zh.shape[0], self.n_heads, self.n_hidden)

        # Compute attention scores 
        e = self.attn(self.activation(g_sum)).squeeze(-1)
        att_score = e.masked_fill(adj_mat == 0, float('-inf'))

        # Apply softmax to normalize the attention scores
        att_score_sm = self.softmax(att_score)
        attn_res = torch.einsum('ijh,jhf->ihf', att_score_sm, g_r)

        return attn_res.mean(dim=1) #, att_score_sm.squeeze(-1)

class GraphRNNCell(nn.Module):
    def __init__(self, in_features, hidden_features, n_heads):
        """
        A single step (cell) of a Graph Recurrent Neural Network using
        GraphAttentionV2Layer + GRUCell.

        Args:
            in_features (int): Number of input features per node.
            hidden_features (int): Number of hidden features per node.
            n_heads (int): Number of attention heads in the GAT layer.
        """
        super(GraphRNNCell, self).__init__()
        # GAT layer for transforming input features
        self.gat = GraphAttentionV2Layer(
            in_features=in_features,
            out_features=hidden_features,
            n_heads=n_heads
        )
        # Recurrent update over hidden state
        self.gru_cell = nn.GRUCell(hidden_features, hidden_features)

    def forward(self, x, adj, h_prev):
        """
        Forward pass of the GraphRNN cell.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_features].
            adj (torch.Tensor): Adjacency matrix of shape [num_nodes, num_nodes].
            h_prev (torch.Tensor): Previous hidden state of shape [num_nodes, hidden_features].

        Returns:
            h_new (torch.Tensor): Updated hidden state of shape [num_nodes, hidden_features].
        """
        # 1) GAT-based transformation of the current node features
        gat_out = self.gat(x, adj)  
        # gat_out -> shape: [num_nodes, hidden_features]

        # 2) Recurrent update (GRUCell) 
        h_new = self.gru_cell(gat_out, h_prev)  
        return h_new

class GraphRNN(nn.Module):
    def __init__(self, in_features, hidden_features, n_heads, num_neurons, n_behaviour):
        """
        Graph Recurrent Neural Network

        Args:
            in_features (int): Number of input features per node.
            hidden_features (int): Number of hidden features per node.
            n_heads (int): Number of attention heads in the GAT layer.
        """
        super(GraphRNN, self).__init__() 
        self.hidden_features = hidden_features               
        self.cell = GraphRNNCell(in_features, hidden_features, n_heads) # The recurrent cell
        # self.readout = nn.Linear(hidden_features, hidden_features)
        self.attention_head = AttentionHead(input_size=hidden_features, hidden_size=128, output_size=n_behaviour, num_neurons=num_neurons)
        # self.dropout = nn.Dropout(p=0.5)  # Apply dropout


    def forward(self, x, adj, h_0=None):
        """  
        Args:
            x (torch.Tensor): Node features. If unrolling over time,
                              shape could be [T, num_nodes, in_features].
                              If single-step, shape could be [num_nodes, in_features].
            adj (torch.Tensor): Adjacency matrix/matrices. Shape depends on
                                whether you have one adjacency matrix or a sequence.
            h_0 (torch.Tensor): Initial hidden state, shape [num_nodes, hidden_features].
                                If None, default to zeros.

        Returns:
            h (torch.Tensor): Final hidden state after ` 
            all_states (torch.Tensor): Hidden states after each step (optional).
        """ 
        T, N, _ = x.shape  
        all_states = []
        z_h = torch.zeros(N, self.hidden_features, device=x.device)
        for t in range(T): 
            z_h = self.cell(x[t], adj.unsqueeze(-1), z_h) 
            all_states.append(z_h)

        # (Optional) pass final hidden state through a readout
        # z_out = self.readout(z_h)  
        rgcn_out = torch.stack(all_states, dim=0) 
        y_pred, neuron_behaviour_alpha  = self.attention_head(rgcn_out) 

        return y_pred, neuron_behaviour_alpha

def load_firing_rates(file_path):
    data = loadmat(file_path)
    for key in ['fr', 'hit', 'miss']:
        if key in data:
            return data[key]
    print(f"None of the keys ('fr', 'hit', 'miss') are available in {file_path}")
    return None

def load_behavior_data(file_path):
    return pd.read_excel(file_path)

def process_data(data_dir, subject, brain_region): 
    data_dir = os.path.join(data_dir, brain_region, subject) 
    neuron_mat_files = glob.glob(os.path.join(data_dir+'/', '*.mat'))    
    
    behavior_data_path = glob.glob(os.path.join(data_dir, brain_region+f'_{subject}.xlsx'))[0]    
    behavior_data = load_behavior_data(behavior_data_path)
    behavior_data[behavior_data.columns[1:]] = behavior_data[behavior_data.columns[1:]].astype(float)
    
    numeric_cols = behavior_data.select_dtypes(include=[float]).columns
    numeric_data_matrix = behavior_data[numeric_cols].values 
 
    data = []
    for i, neuron_i_path in enumerate(neuron_mat_files): 
        neuron_i_data = load_firing_rates(neuron_i_path) 
        neuron_i_data = torch.tensor(neuron_i_data).unsqueeze(0)
        data.append(neuron_i_data) 
    data = torch.cat(data) 
    data = rearrange(data, 'neuron trail signal -> trail neuron signal')
     
    return data.float(), torch.tensor(numeric_data_matrix), numeric_cols

def calculate_adjacency_matrix(spatiotemporal_tensor):
    """
    Calculates the adjacency matrix based on the correlation of the 'time' dimension of a spatiotemporal tensor
    using 1 - squareform(pdist(..., 'correlation')).

    :param spatiotemporal_tensor: A 2D numpy array of shape [node, time]
    :return: A 2D numpy array representing the adjacency matrix of shape [node, node]
    """ 
    adjacency_matrix = 1 - squareform(pdist(spatiotemporal_tensor, 'correlation'))
    return torch.tensor(adjacency_matrix)

def calculate_r2(train_pred: torch.Tensor, train_gt: torch.Tensor) -> torch.Tensor:
    """
    Calculate the R² (coefficient of determination) for each behavior.

    Args:
        train_pred (torch.Tensor): Predicted values, shape [trials, neurons, behaviors].
        train_gt (torch.Tensor): Ground truth values, shape [trials, neurons, behaviors].

    Returns:
        torch.Tensor: R² values for each behavior, shape [behaviors].
    """
    # Calculate the mean of ground truth across trials
    mean_train_gt = train_gt.mean(dim=0)  # Shape: [neurons, behaviors]

    # Calculate the residual sum of squares (SS_residual)
    ss_residual = torch.sum((train_gt - train_pred) ** 2, dim=0)  # Shape: [neurons, behaviors]

    # Calculate the total sum of squares (SS_total)
    ss_total = torch.sum((train_gt - mean_train_gt) ** 2, dim=0)  # Shape: [neurons, behaviors]

    # Handle cases where SS_total is zero to avoid division by zero
    with torch.no_grad():
        ss_total = torch.where(ss_total == 0, torch.tensor(float("inf"), device=ss_total.device), ss_total)

    # Calculate R²
    r2 = 1 - (ss_residual / ss_total)  # Shape: [neurons, behaviors]

    # Average R² across neurons for each behavior
    r2_per_behavior = r2.mean(dim=0)  # Shape: [behaviors]

    return r2_per_behavior

def create_dataloaders(X, Y, train_size_perc, val_size_perc, batch_size):
    """

    Create train, validation, and test DataLoaders.

    Args:
        X (torch.Tensor): The dataset features.
        Y (torch.Tensor): The behavior labels.
        train_size_perc (int): Percentage of data for training.
        val_size_perc (int): Percentage of data for validation.
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Shuffle and split the dataset into train, validation, and test sets        
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_size = int((train_size_perc/100) * len(X))
    val_size = int((val_size_perc/100) * len(X))
    
    # Split into train, validation, test sets
    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]
    
    # Create datasets and DataLoader objects (you can define batch_size=batch_size or batch_size=len(X[train_indices]))
    train_loader = DataLoader(TensorDataset(X[train_indices].float().to(device), Y[train_indices].float().to(device)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X[val_indices].float().to(device), Y[val_indices].float().to(device)), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X[test_indices].float().to(device), Y[test_indices].float().to(device)), batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

def plot_rnn_metrics(train_r2, val_r2, test_r2, train_loss, val_loss, test_loss, y_label, epochs, brain_region, subject, behav, save_path):
    """
    Plot R² and loss metrics for training, validation, and test phases of an RNN model.
    
    Parameters:
        train_r2 (list of tensors): R² values for training (per behavior, per epoch).
        val_r2 (list of tensors): R² values for validation (per behavior, per epoch).
        test_r2 (tensor): R² values for test (per behavior).
        train_loss (tensor): Loss values for training (per epoch).
        val_loss (tensor): Loss values for validation (per epoch).
        test_loss (float): Test loss (single value).
        y_label (list): List of behaviors.
        epochs (int): Number of epochs.
        brain_region (str): Name of the brain region.
        subject (str): Subject identifier.
        save_path (str): Directory to save the plot.
    """
    # Mean
    mean_train_r2 = np.mean(np.array(train_r2), axis=0)
    mean_val_r2 = np.mean(np.array(val_r2), axis=0)
    mean_test_r2 = np.mean(np.array(test_r2), axis=0)
    mean_train_loss = np.mean(np.array(train_loss), axis=0)
    mean_val_loss = np.mean(np.array(val_loss), axis=0)
    mean_test_loss = np.mean(np.array(test_loss), axis=0)

    # Bootstrapping
    train_r2_l_ci, train_r2_u_ci = bootstrap_ci_gpu(train_r2, n_iterations=1000, ci=95)
    val_r2_l_ci, val_r2_u_ci = bootstrap_ci_gpu(val_r2, n_iterations=1000, ci=95)
    test_r2_l_ci, test_r2_u_ci = bootstrap_ci_gpu(test_r2, n_iterations=1000, ci=95)
    train_loss_l_ci, train_loss_u_ci = bootstrap_ci_gpu(train_loss, n_iterations=1000, ci=95)
    val_loss_l_ci, val_loss_u_ci = bootstrap_ci_gpu(val_loss, n_iterations=1000, ci=95)
    test_loss_l_ci, test_loss_u_ci = bootstrap_ci_gpu(test_loss, n_iterations=1000, ci=95)

    # Convert behaviors to a list if not already
    behaviors = list(y_label)
    colors = plt.cm.tab10(np.linspace(0, 1, len(behaviors)))

    # Create a figure and define a gridspec layout
    fig = plt.figure(figsize=(18, 15), frameon=False)
    gs = gridspec.GridSpec(3, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.1])

    # Training R² with bootstrap CI
    ax_r2_train = fig.add_subplot(gs[0, :3])
    for i, behavior in enumerate(behaviors):
        behavior_r2 = [mean_train_r2[epoch][i] for epoch in range(epochs)]
        l_ci = [train_r2_l_ci[epoch][i] for epoch in range(epochs)]
        u_ci = [train_r2_u_ci[epoch][i] for epoch in range(epochs)]
        ax_r2_train.plot(range(epochs), behavior_r2, label=f'{behavior}', color=colors[i])
        ax_r2_train.fill_between(range(epochs), l_ci, u_ci, color=colors[i], alpha=0.2)
    ax_r2_train.set_title('Training R²')
    ax_r2_train.set_xlabel('Epoch')
    ax_r2_train.set_ylabel('R²')
    ax_r2_train.grid()

    # Training Loss with bootstrap CI
    ax_loss_train = fig.add_subplot(gs[0, 3:])    
    ax_loss_train.plot(range(epochs), mean_train_loss, label='Train Loss', color='red')
    ax_loss_train.fill_between(range(epochs), train_loss_l_ci.flatten(), train_loss_u_ci.flatten(), color='red', alpha=0.2)
    ax_loss_train.set_title('Training Loss')
    ax_loss_train.set_xlabel('Epoch')
    ax_loss_train.set_ylabel('Loss')
    ax_loss_train.grid()

    # Validation R² with bootstrap CI
    ax_r2_val = fig.add_subplot(gs[1, :3])
    for i, behavior in enumerate(behaviors):
        behavior_r2 = [mean_val_r2[epoch][i] for epoch in range(epochs)]
        l_ci = [val_r2_l_ci[epoch][i] for epoch in range(epochs)]
        u_ci = [val_r2_u_ci[epoch][i] for epoch in range(epochs)]
        ax_r2_val.plot(range(epochs), behavior_r2, label=f'{behavior}', color=colors[i])
        ax_r2_val.fill_between(range(epochs), l_ci, u_ci, color=colors[i], alpha=0.2)
    ax_r2_val.set_title('Validation R²')
    ax_r2_val.set_xlabel('Epoch')
    ax_r2_val.set_ylabel('R²')
    ax_r2_val.grid()

    # Validation Loss with bootstrap CI
    ax_loss_val = fig.add_subplot(gs[1, 3:])
    ax_loss_val.plot(range(epochs), mean_val_loss, label='Validation Loss', color='red')
    ax_loss_val.fill_between(range(epochs), val_loss_l_ci.flatten(), val_loss_u_ci.flatten(), color='red', alpha=0.2)
    ax_loss_val.set_title('Validation Loss')
    ax_loss_val.set_xlabel('Epoch')
    ax_loss_val.set_ylabel('Loss')
    ax_loss_val.grid()

    # Test R² with bootstrap CI
    ax_r2_test = fig.add_subplot(gs[2, :2])
    ax_r2_test.scatter(range(len(mean_test_r2[0])), mean_test_r2.flatten(), color=colors[:len(mean_test_r2[0])], alpha=1, marker='_', s=800)
    ax_r2_test.errorbar(range(len(mean_test_r2[0])), mean_test_r2.flatten(), yerr=[mean_test_r2.flatten() - test_r2_l_ci.flatten(), test_r2_u_ci.flatten() - mean_test_r2.flatten()], fmt='none', color='black', capsize=5)
    ax_r2_test.set_title('Test R²')
    # ax_r2_test.set_xlabel('Behaviors')
    ax_r2_test.set_ylabel('R²')
    ax_r2_test.set_xticks(range(len(mean_test_r2[0])))
    ax_r2_test.set_xticklabels(behaviors, rotation=45, ha='right')
    ax_r2_test.grid()

    # Test Loss with bootstrap CI
    ax_loss_test = fig.add_subplot(gs[2, 2:4])
    ax_loss_test.scatter(0, mean_test_loss.flatten(), label='Test Loss', color='red', alpha=1, marker='_', s=800)
    ax_loss_test.errorbar(0, mean_test_loss.flatten(), yerr=[mean_test_loss.flatten() - test_loss_l_ci.flatten(), test_loss_u_ci.flatten() - mean_test_loss.flatten()], fmt='none', color='black', capsize=5)
    ax_loss_test.set_title('Test Loss')
    ax_loss_test.set_ylabel('Loss')
    ax_loss_test.set_xticks([])
    ax_loss_test.grid()

    # Legend
    ax_legend = fig.add_subplot(gs[2, 4:])
    for i, behavior in enumerate(behaviors):
        ax_legend.bar(0, 0, label=behavior, color=colors[i])
    ax_legend.legend(loc='center left', frameon=False, fontsize=10, ncol=1)
    ax_legend.axis('off')

    plt.tight_layout()

    # Save the figure
    file_name = f"{brain_region}_{subject}_{behav}.png"
    plt.savefig(os.path.join(save_path, file_name), format="png")
    # plt.show()
    plt.close(fig)

# Perform bootstrap sampling and calculate confidence intervals for R² or loss
def bootstrap_ci_gpu(data, n_iterations=1000, ci=95):
    # Move data to GPU
    data = torch.tensor(data, device='cuda')  # Assuming `data` is a NumPy array
    n_rows, n_cols, n_conditions = data.shape  # Extract dimensions
    
    lower_ci = torch.zeros(n_cols, n_conditions, device='cuda')
    upper_ci = torch.zeros(n_cols, n_conditions, device='cuda')
    
    for col in range(n_cols):  # Loop over the 75 samples (axis 1)
        for cond in range(n_conditions):  # Loop over the 6 conditions (axis 2)
            column_data = data[:, col, cond]
            means = torch.zeros(n_iterations, device='cuda')
            
            for i in range(n_iterations):
                # Perform sampling with replacement across axis 0 (subjects)
                sample_indices = torch.randint(0, n_rows, (n_rows,), device='cuda')
                sample = column_data[sample_indices]
                means[i] = sample.mean().detach()

            # Calculate the confidence intervals for this behavior and condition
            lower_ci[col, cond] = torch.quantile(means, (100 - ci) / 2 / 100).item()
            upper_ci[col, cond] = torch.quantile(means, 1 - (100 - ci) / 2 / 100).item()
    
    # Move results back to CPU for further processing or return them
    return lower_ci.cpu().numpy(), upper_ci.cpu().numpy()

# Clear Spyder console
def clear_console():
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic('clear')

#%% PROGRAM STARTS HERE
clear_console()  # Clear the Spyder console at the beginning of each epoch
# Define hyperparameters
n_gcn_heads = 4
n_hidden_features = 24 # 12, 18, (24)
epochs = 75
batch_size = 64 # (64)
learn_rate = 1e-3 # 1.5e-4 7.5e-4(24)
brain_region    = 'ACC' # 'ACC', 'CLA'
subject         = 'sub016A' #
rounds = 2 # number of rounding training

# SUBJECTS
# ACC - 'sub016A', 'sub016B', 'sub019A', 'sub019B', 'sub020', 'sub023', sub024A'
# CLA - 'sub016A', 'sub016B', 'sub017', 'sub024A'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device configuration (CPU or GPU)

#%%
# Load Data where x - Data, y - Behavioral Data
# data_dir = r'C:\Users\rd883\Desktop\maxd\Spaceship\Figures\Data\RNN'
data_dir = '/gpfs/radev/home/aa2793/project/brainnn/dataset/brainpatterner/ieeg/' 
x, Y, Y_label = process_data(data_dir, subject, brain_region)
# import pdb;pdb.set_trace()
# Remove Behaviors
# indices_to_remove = list(range(1, 5)) + list(range(13, 19)) # Identify the column indices to remove
indices_to_remove = list(range(0, 3)) + list(range(5, 7))  + list(range(9, 19)) # Identify the column indices to remove

all_indices = list(range(Y.size(1)))  # Create a mask to select the columns you want to keep
indices_to_keep = [i for i in all_indices if i not in indices_to_remove] # Columns to keep
Y = Y[:, indices_to_keep] # Index the tensor to keep only the desired columns
Y_label = Y_label[indices_to_keep]
    
# for n_behav in range(len(Y[0])): # Training each Behavior
#     y_label = pd.Index([Y_label[n_behav]])    
#     y = Y[:,n_behav].view(-1,1) # if you want to apply Graph RNN per Behavior, you need to define which behavior
#     y = (y - y.min()) / (y.max() - y.min()) # Normalize Min-Max [0 - 1]
#     # y = y - y.mean() # Demeaning
#     # y = (y - y.mean()) / y.std() # Z-Scored
    
for n_behav in range(0, 1): # Training all Behaviors
    y_label = pd.Index(Y_label)  
    y = Y
    y = (y - y.min(dim=0).values) / (y.max(dim=0).values - y.min(dim=0).values)

    # Normalize Firing Rates and Behaviors
    x = (x - x.min()) / (x.max() - x.min()) # Normalize Min-Max [0 - 1]
    # x = x - x.mean(dim=1, keepdim=True) # Demeaning across time
    # x = (x - x.mean()) / x.std() # Z-Scored  
    
    # Initialize arrays to store results
    train_r2_all = []
    val_r2_all = []
    test_r2_all = []
    train_loss_all = []
    val_loss_all = []
    test_loss_all = []
    # Training and Test "rounds" times
    for i in range(rounds):
        clear_console()
        # increase training set to 90, validation to 5 (or 1), and test to 5 (or 9)
        train_data, val_data, test_data = create_dataloaders(x, y, train_size_perc=65, val_size_perc=15, batch_size=batch_size)
        # Load Adjacency Matrix
        adj_matrix = calculate_adjacency_matrix(rearrange(train_data.dataset.tensors[0], 'trails nodes time -> nodes (trails time)').cpu().numpy())
        adj_matrix = adj_matrix.to(device)
        # Initialize model, Loss, and Optimizer
        model = GraphRNN(in_features=x.shape[2], hidden_features=n_hidden_features, n_heads=n_gcn_heads, num_neurons=x.shape[1], n_behaviour=y.shape[1]) 
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    
        best_val_loss = float('inf') # Initialize best validation loss with a high value
    
        # Training loop

        train_r2 = []  # To store the sequence of val_r2 values
        val_r2 = []
        test_r2 = []
        train_loss = []
        val_loss = []
        test_loss = []
        best_val_r2 = []  # To store the sequence of val_r2 values
        best_val_r2 = None  # Placeholder for best R²
        best_val_loss = float('inf')  # Start with infinity for loss comparison
        for epoch in range(epochs):  # Define num_epochs or specify a training criterion
            tr_loss = 0
            vl_loss = 0
            te_loss = 0
            r2 = []
            model.train()
            for train_inputs, train_labels in train_data:
                optimizer.zero_grad()
                train_pred, neuron_behaviour_alpha_train = model(train_inputs, adj_matrix)
                train_gt = train_labels.unsqueeze(1).repeat(1, train_pred.shape[1], 1)
                loss = criterion(train_pred, train_gt).float()  # Adjust loss to match your label shape     
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)                    
                r2.append(calculate_r2(train_pred, train_gt))  # Append the current train_r2 to the history 
                
            tr_loss /= len(train_data)
            train_loss.append(tr_loss)
            r2 = torch.stack(r2)  # Stack the 12 tensors into a single tensor of shape (12, 7)
            r2 = r2.mean(dim=0)  # Take the mean across the first dimension (the 12 tensors)
            train_r2.append(r2)  # Append the current train_r2 to the history 
            
            # Validation loop
            model.eval()
            r2 = []
            with torch.no_grad():  # Disable gradient tracking for validation
                for val_inputs, val_labels in val_data:
                    val_pred, neuron_behaviour_alpha_val = model(val_inputs, adj_matrix)
                    val_gt = val_labels.unsqueeze(1).repeat(1, val_pred.shape[1], 1)
                    loss = criterion(val_pred, val_gt).float()
                    vl_loss += loss.item()
                    r2.append(calculate_r2(val_pred, val_gt))
                    
                    # Check if this is the best model based on validation loss
                    if loss < best_val_loss:
                        best_val_loss = loss
                        best_model_params = {k: v.clone() for k, v in model.state_dict().items()}
                        best_val_r2 = calculate_r2(val_pred, val_gt) # Store the R² of the best model
    
            vl_loss /= len(val_data)
            scheduler.step(vl_loss)
            val_loss.append(vl_loss)
            r2 = torch.stack(r2)  # Stack the 12 tensors into a single tensor of shape (12, 7)
            r2 = r2.mean(dim=0)  # Take the mean across the first dimension (the 12 tensors)
            val_r2.append(r2)  # Append the current train_r2 to the history 
            
            # Optionally, print results
            if epoch % 10 == 0:
                print(f"Training Run {i+1} of {rounds}, Epoch {epoch}/{epochs}, Train Loss: {tr_loss}, Validation Loss: {vl_loss}")
        
        # Testing loop
        model.load_state_dict(best_model_params) # load the best model parameters for testing
        model.eval()
        r2 = []
        with torch.no_grad():
            for test_inputs, test_labels in test_data:
                test_pred, neuron_behaviour_alpha_test = model(test_inputs, adj_matrix)
                test_gt = test_labels.unsqueeze(1).repeat(1, test_pred.shape[1], 1)
                loss = criterion(test_pred, test_gt).float()
                te_loss += loss.item()
                r2.append(calculate_r2(test_pred, test_gt))
        
            te_loss /= len(test_data)
            test_loss.append(te_loss)
            r2 = torch.stack(r2)
            r2 = r2.mean(dim=0)
            test_r2.append(r2)
        print(f"Test Loss: {np.mean(test_loss)}")
    
        # Store results for this run
        train_r2_all.append([tensor.detach().cpu().numpy() for tensor in train_r2])
        val_r2_all.append([tensor.detach().cpu().numpy() for tensor in val_r2])
        test_r2_all.append([tensor.detach().cpu().numpy() for tensor in test_r2])
        train_loss_all.append(np.array(train_loss))
        val_loss_all.append(np.array(val_loss))
        test_loss_all.append(np.array(test_loss))

# Convert results to numpy arrays
train_r2_all = np.array(train_r2_all)
val_r2_all = np.array(val_r2_all)
test_r2_all = np.array(test_r2_all)
train_loss_all = np.array(train_loss_all)
val_loss_all = np.array(val_loss_all)
test_loss_all = np.array(test_loss_all)
# add [:,:,1]
train_loss_all = np.expand_dims(train_loss_all, axis=-1)
val_loss_all = np.expand_dims(val_loss_all, axis=-1)
test_loss_all = np.expand_dims(test_loss_all, axis=-1)

#%% Plot and Save
if len(y[0]) == 1:
    plot_rnn_metrics(train_r2_all, val_r2_all, test_r2_all, train_loss_all, val_loss_all, test_loss_all, y_label, epochs, brain_region, subject, y_label[0], r'C:\Users\rd883\Desktop\Graph RNN')
else:
    plot_rnn_metrics(train_r2_all, val_r2_all, test_r2_all, train_loss_all, val_loss_all, test_loss_all, y_label, epochs, brain_region, subject, 'All', r'C:\Users\rd883\Desktop\Graph RNN')
    
