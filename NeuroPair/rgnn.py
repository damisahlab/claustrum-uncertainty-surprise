 
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
from einops import rearrange
from scipy.io import loadmat
import pandas as pd
from scipy.spatial.distance import pdist, squareform 
import torch.nn.functional as F
# from IPython import get_ipython
from sklearn.model_selection import train_test_split
import numpy as np
 

from data.adj_matrix import calculate_adjacency_matrix
from nn.gat import GraphAttentionV2Layer
from nn.lstm import RnnLSTM 
from nn.gat import AttentionHead

from model.recurrentgat import RecurrentGAT
 
epochs = 200 
brain_region    = 'ACC' # 'ACC', 'CLA', 'INS'
subject         = 'sub016A'

device = 'cuda'
data_dir = '/gpfs/radev/home/aa2793/project/brainnn/dataset/brainpatterner/ieeg/' 


from data.dataloader import process_data
x, y, y_label = process_data(data_dir, subject, brain_region) 

 


indices_to_remove = list(range(1, 5)) + list(range(13, 19))  
all_indices = list(range(y.size(1)))   
indices_to_keep = [i for i in all_indices if i not in indices_to_remove]  
y = y[:, indices_to_keep]  
y_label = y_label[indices_to_keep]

 

rounds = 1
for i in range(rounds): 

    x = x[:, :, :120]  # drop the last one to be devidable to train/test/validation
    total_size = x.shape[2]   
    train_indices = torch.arange(0, total_size, 3) 
    val_indices = torch.arange(1, total_size, 3) 
    test_indices = torch.arange(2, total_size, 3) 

    train_data = x[:, :, train_indices] 
    val_data = x[:, :, val_indices] 
    test_data = x[:, :, test_indices] 

    seq_length, num_neurons, num_samples = train_data.shape

    adj_matrix = calculate_adjacency_matrix(rearrange(train_data, 'trails nodes time -> nodes (trails time)'))   
    test_data = x[:, :, test_indices].float()
    
    model = RecurrentGAT(
        rnn_num_layers = 2, input_size = num_samples, hidden_size  = 128, dropout = 0.2, gat_n_heads = 4,
     num_neurons=num_neurons, n_behaviour=y.shape[1], device=device)
    model.to(device) 
    
    n_parameters = np.sum([p.numel() for p in model.parameters()])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf') # Initialize best validation loss with a high value

    # Training loop
    model.train()
    y = y.unsqueeze(1).repeat(1, x.shape[1], 1).float().to(device)
    for epoch in range(epochs):   
        # Forward pass for training data
        y_pred, neuron_behaviour_alpha_train = model(train_data.to(device), adj_matrix.to(device)) 
        train_loss = criterion(y_pred, y) 

          
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            y_pred, neuron_behaviour_alpha_val = model(val_data.to(device), adj_matrix.to(device)) 
            val_loss = criterion(y_pred, y) 
            # Check if this is the best model                   
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_params = {k: v.clone() for k, v in model.state_dict().items()}     
        
        # Optionally, print results
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")

    # Testing loop
    model.load_state_dict(best_model_params) # load the best model parameters for testing
    model.eval()
    with torch.no_grad():
        test_pred, neuron_behaviour_alpha_test = model(test_data, adj_matrix)
        test_gt = test_behav.unsqueeze(1).repeat(1, test_pred.shape[1], 1)
        test_loss = criterion(test_pred, test_gt).float()
        print(f"Test Loss: {test_loss.item()}")