#%% Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import shutil
import pickle

from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from einops import rearrange

from data.dataloader import process_data
from data.datasplit import dataset_splitter
from model import model_registry
from args import argparser_fn, print_gpu_info, make_directroy
from viz.scatter import scatter_plot

#%% Training Functions
def one_epoch_train(args, model, optimizer, train_data, adj_matrix):
    """
    Performs one epoch of training.
    Adds noise if noise regularization is enabled.
    """ 
    (x_train, y_train) = train_data
    
    # Add Gaussian noise if requested
    if args.noise_reg:
        std_x_train = torch.std(x_train, dim=0) 
        noise = args.noise_th / 100 * std_x_train * torch.randn(x_train.size()).to(args.device) 
        x_train = x_train + noise

    model.train()   
    optimizer.zero_grad()     
    
    y_hat, alphas, z = model(x_train, adj_matrix)  
    loss = model.loss(y_train, y_hat, reduction='sum')  
    loss.backward()                  
    optimizer.step()   
    return loss.item(), alphas, y_hat, z

def one_epoch_validation(args, model, data, adj_matrix):
    """
    Evaluates the model on validation or test data.
    """
    (x, y) = data
    model.eval()    
    with torch.no_grad(): 
        y_hat, alphas, z = model(x, adj_matrix)      
        loss = model.loss(y, y_hat, reduction='sum')    
    return loss.item(), alphas, y_hat, z
 
def extract_interest_behaviour(y_all, y_labels_all, interested_behaviours): 
    """
    Extracts target behaviors of interest from all available labels.
    Returns the corresponding y tensor and label list.
    """
    y, y_labels = [], []
    for i, y_label in enumerate(y_labels_all):
        if y_label in interested_behaviours: 
            y.append(y_all[:, i].unsqueeze(-1))
            y_labels.append(y_label)
    y = torch.cat(y, dim=-1)  
    return y, y_labels

def train(args, interested_behaviours, repeat_index):
    """
    Trains the model for a given set of behaviors and saves outputs.
    """
    # Load and process data
    x, y_all, y_labels_all = process_data(args.data_dir, args.subject_id, args.brain_region)
    y, y_label = extract_interest_behaviour(y_all, y_labels_all, interested_behaviours) 
    
    # Split dataset
    args, train_data, val_data, test_data, adj_matrix = dataset_splitter(args, x, y, y_label) 
    
    # Initialize model
    args.num_neurons = train_data[0].shape[1]
    model = model_registry.str2model(args.model)  
    model = model(args).to(args.device)
    
    # Set up output directories and save a copy of the code
    args.output_dir, key_save_dir = make_directroy(args, interested_behaviours[0], repeat_index)   
    shutil.copytree('.', args.output_dir+'/codes/', dirs_exist_ok=True)   
    
    best_val_loss = float('inf') 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
    # Inner trail training (Training loop)
    for epoch in range(args.num_epochs):
        train_loss, train_alpha, train_y_hat, train_z = one_epoch_train(
            args,
            model,
            optimizer,
            train_data,
            adj_matrix
        )

        val_loss, val_alpha, val_y_hat, val_z = one_epoch_validation(
            args,
            model,
            val_data,
            adj_matrix,
        )

        # Undo any target transformation
        val_y_hat = model.untransform_targets(
            val_y_hat.squeeze(-1)).unsqueeze(-1
        )
        
        # Save current and best models
        torch.save(model, args.output_dir+'/model/current_model.pth') 
        if val_loss < best_val_loss:
            best_val_loss = val_loss  
            torch.save(model, args.output_dir+'/model/best_model.pth') 

        # Compute performance metrics (R^2 value)
        y = val_data[1].flatten().cpu().numpy()
        y_pred = val_y_hat.flatten().cpu().numpy()

        r2_val = r2_score(y, y_pred)
        spearman_corr, _ = spearmanr(y, y_pred)

        # Print training progress
        print(
            f"Running epoch {epoch + 1}/{args.num_epochs}, "
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Training Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"R2: {r2_val:.4f}, "
            f"Spearman: {spearman_corr:.4f}"
        )

        # Save scatter plots every 100 epochs
        if epoch % 100 == 0:
            scatter_plot(
                x_ut=y,
                y_ut=y_pred,
                path=f"{args.output_dir}/scatter_plot_validation/scatter_{epoch}",
                title="",
                label=None,
                dpi=100,
                scatter_color='#0077b6',
                title_add=True
            )
    return model

def all_sub_region_train(repeat_index, subject_id_list, brain_region_list):
    """
    Iterates over subjects, brain regions, and behaviors to train and save models.
    """
    for subject_id in subject_id_list:
        # Initialize dictionaries to track models
        model_key_acc = {
            'A_prediction_error':'', 
            'B_prediction_error':'', 
            'A_safety_variance': '',
            'B_safety_variance': '', 
        }

        model_key_cla = {
            'A_prediction_error':'', 
            'B_prediction_error':'', 
            'A_safety_variance': '',
            'B_safety_variance': '', 
        }
   
        for brain_region in brain_region_list: 
            for interested_behavior in interested_behaviours_list: 
                args = argparser_fn(subject_id=subject_id, brain_region=brain_region, dataset_type='ieeg') 
                # Track model output keys
                if brain_region == 'ACC':
                    model_key_acc[interested_behavior] = args.output_key 
                else:
                    model_key_cla[interested_behavior] = args.output_key 
                
                # args.output_key = subject_id+'--'+ args.output_key
                print_gpu_info(args.output_key)   
                # Train Model
                args.output_dir, key_save_dir = make_directroy(args, interested_behavior, repeat_index)    
                model = train(args, [interested_behavior], repeat_index) 
                # Save keys
                with open(key_save_dir+"/keys/model_key_acc.pkl", "wb") as file:  
                    pickle.dump(model_key_acc, file) 
                with open(key_save_dir+"/keys/model_key_cla.pkl", "wb") as file: 
                    pickle.dump(model_key_cla, file)

#%% Main Script
if __name__ == "__main__":  
    subject_id_list = ['sub016A', 'sub016B', 'sub024A']  
    onlyACC_subject_id_list = ['sub019A', 'sub019B', 'sub020', 'sub023']
    onlyCLA_subject_id_list = ['sub017_CLA']  
    repeat_index = '10'

    interested_behaviours_list = [
        'A_prediction_error',
        'B_prediction_error',
        'A_safety_variance', 
        'B_safety_variance', 
    ] 


    # Train across all subjects and regions
    all_sub_region_train(repeat_index, subject_id_list, ['ACC', 'CLA'], interested_behaviours_list)
    all_sub_region_train(repeat_index, onlyACC_subject_id_list, ['ACC'], interested_behaviours_list)
    all_sub_region_train(repeat_index, onlyCLA_subject_id_list, ['CLA'], interested_behaviours_list)