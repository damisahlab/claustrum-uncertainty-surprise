from data.dataloader import process_data 
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from args import argparser_fn, print_gpu_info, make_directroy
from data.adj_matrix import calculate_adjacency_matrix
from einops import rearrange 
from model import model_registry
import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from viz.scatter import scatter_plot
 
def one_epoch_train(args, model, criterion, optimizer, train_data, adj_matrix):
    (x_train, y_train) = train_data

    # Calculate the standard deviation of x_train
    std_x_train = torch.std(x_train, dim=0)

    # Generate noise of th% of the standard deviation
    noise = args.noise_th / 100 * std_x_train * torch.randn(x_train.size()).to(args.device)

    # Add noise to x_train
    x_train = x_train + noise

    model.train()   
    optimizer.zero_grad()     

    # Use noisy_x_train in the model
    y_hat, alphas, z = model(x_train, adj_matrix)  
    loss = model.loss(y_train, y_hat, reduction='sum')  
    loss.backward()                  
    optimizer.step()   
    return loss.item(), alphas, y_hat, z



def one_epoch_validation(args, model, criterion, data, adj_matrix): 
    (x, y) = data
    model.eval()    
    with torch.no_grad(): 
        y_hat, alphas, z = model(x, adj_matrix)        
        # loss = criterion(y_hat, y)
        loss = model.loss(y, y_hat, reduction='sum')   
    return loss.item(), alphas, y_hat, z
 

 

import torch
import matplotlib.pyplot as plt
from einops import rearrange  

def plot_tensor(x):
    # Calculate the mean over the first dimension to reduce the tensor to a [21, 120] matrix
    # x_avg = rearrange(x, 'a b c -> b (a c)') 

    # Set up the plot
    for signal in range(x.shape[0]):
    
        x_avg = x[signal, :, :]
        fig, ax = plt.subplots(figsize=(20, 6))
        
        # Create a color map
        cmap = plt.get_cmap('viridis', x_avg.shape[0])

        # Plot each line with a different color
        for i in range(x_avg.shape[0]):
            ax.plot(x_avg[i].numpy(), color=cmap(i), label=f'Slice {i+1}')  # Convert to numpy for plotting

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, x_avg.shape[0] - 1))
        sm.set_array([])  # You can set an empty array or pass the actual data
        cbar = plt.colorbar(sm, ax=ax, ticks=range(x_avg.shape[0]), label='Slice Index')
        cbar.set_label('Slice Index')

        # ax.set_xlabel('Dimension 3 (length 120)')
        # ax.set_ylabel('Average Value')
        # ax.set_title('Plot of Tensor by Last Dimension, Colored by Second Dimension')
        ax.legend(title='Slice Index', bbox_to_anchor=(1.05, 1), loc='upper left') 
        plt.savefig('../outputs/brain_patterner/trails/x_trail_'+str(signal)+'.png')
        plt.close()
        if signal==5: break

def prepare_data(args, x, y, y_label): 
    # x_seed = x[:, :, 0].unsqueeze(-1)
    x = x[:, :, 1:]  # drop the first for feasibility of the trian/test and validation splits
    # import pdb;pdb.set_trace()
    # plot_tensor(x)
    # x = F.normalize(x, p=2, dim=1) 
    # y = F.normalize(y, p=0, dim=-1)
    # Assuming x and y are your tensors
    # x = x[:, :, 1:]  # drop the first for feasibility of the train/test and validation splits
    # x_min = x.min(dim=-1, keepdim=True)[0]
    # x_max = x.max(dim=-1, keepdim=True)[0]
    # x = (x - x_min) / (x_max - x_min)

    # Similarly for y, assuming y is a tensor that needs to be scaled
    # y_min = y.min(dim=-1, keepdim=True)[0]
    # y_max = y.max(dim=-1, keepdim=True)[0]
    # y = (y - y_min) / (y_max - y_min)  
 
    total_size = x.shape[2]   
    train_indices = torch.arange(0, total_size, 2) 
    val_indices = torch.arange(1, total_size, 2)  

    x_train = x[:, :, train_indices] 
    x_val = x[:, :, val_indices] 
    x_test = x_val

    # train_indices = torch.arange(0, total_size, 3) 
    # val_indices = torch.arange(1, total_size, 3) 
    # test_indices = torch.arange(2, total_size, 3) 

    # x_train = x[:, :, train_indices] 
    # x_val = x[:, :, val_indices] 
    # x_test = x[:, :, test_indices] 

    seq_length, num_neurons, num_samples = x_train.shape   
    adj_matrix = calculate_adjacency_matrix(rearrange(x_train, 't n s -> n (t s)'))   
    adj_matrix = adj_matrix.to(args.device)

    args.n_behaviour = y.shape[1]
    args.num_neurons = num_neurons
    args.input_size = num_samples  
 
    # y_train = y.unsqueeze(1).float().to(args.device)
    # y_val = y.unsqueeze(1).float().to(args.device)
    # y_test = y.unsqueeze(1).float().to(args.device)
    y_train = y.unsqueeze(1).repeat(1, x_train.shape[1], 1).float().to(args.device)
    y_val = y.unsqueeze(1).repeat(1, x_val.shape[1], 1).float().to(args.device)
    y_test = y.unsqueeze(1).repeat(1, x_test.shape[1], 1).float().to(args.device)

    x_train = x_train.to(args.device) 
    x_val = x_val.to(args.device) 
    x_test = x_test.to(args.device) 

    return args, (x_train, y_train), (x_val, y_val), (x_test, y_test), adj_matrix
 


if __name__ == "__main__":  
    batch_size = 12 

    dataset_type = 'ieeg'   
    args = argparser_fn(dataset_type, batch_size) 
    print_gpu_info(args.output_key)
 
    brain_region = 'CLA' # 'ACC', 'CLA', 'INS'
    subject = 'sub016A'

    device = 'cuda'
    data_dir = '/gpfs/radev/home/aa2793/project/brainnn/dataset/brainpatterner/ieeg/' 
    interested_behaviours = ['A_safety_value', 'B_safety_value']

    from data.dataloader import process_data
    x, y_all, y_labels_all = process_data(data_dir, subject, brain_region)  

    y, y_labels = [], []
    for i, y_label in enumerate(y_labels_all):
        if y_label in interested_behaviours:
            y.append(y_all[:, i].unsqueeze(-1))
            y_labels.append(y_label)

    y = torch.cat(y, dim=-1)


    
    # from viz.histogram import plot_histograms
    # plot_histograms(y, y_labels)

    args, train_data, val_data, test_data, adj_matrix = prepare_data(args, x, y, y_label)
    model = model_registry.str2model(args.model)  
    model = model(args).to(args.device)  

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    run_loss, trial_r2_val = [], [] 
    best_val_loss = float('inf') 
    
    # Inner trail training
    for epoch in range(args.num_epochs):
        train_loss, train_alpha, train_y_hat, train_z  = one_epoch_train(
            args, 
            model, 
            criterion, 
            optimizer,  
            train_data, 
            adj_matrix
        )
        
        val_loss, val_alpha, val_y_hat, val_z = one_epoch_validation(
            args, 
            model, 
            criterion, 
            val_data, 
            adj_matrix,
        )
                             
        if val_loss < best_val_loss:
            best_val_loss = val_loss  
 
        # Compute R^2 value
        y = val_data[1].flatten().cpu().numpy()
        y_pred = val_y_hat.flatten().cpu().numpy()

        r2_val = r2_score(y, y_pred) 
        spearman_corr, _ = spearmanr(y, y_pred) 

        print(f"Running epoch {epoch + 1}/{args.num_epochs}, "
                f"Epoch {epoch + 1}/{args.num_epochs} - "
                f"Training Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"R2: {r2_val:.4f}",
                f"Spearman: {spearman_corr:.4f}",
        ) 
                
        trial_r2_val.append(r2_val)   
        run_loss.append(best_val_loss) 
        
        if epoch % 100 ==0: 
            scatter_plot(
                x_ut=y, 
                y_ut=y_pred, 
                path="../outputs/brain_patterner/val-scatter/epoch-"+str(epoch), 
                title="", 
                label=None, 
                dpi=50, 
                scatter_color='#0077b6', 
                title_add=True
            )


test_loss, val_alpha, val_y_hat, val_z = one_epoch_validation(args, model, criterion, test_data, adj_matrix)  
actual = test_data[1].flatten().cpu().numpy()
predicted = test_y_hat.flatten().cpu().numpy()  
r2_test = r2_score(actual, predicted) 
spearman_corr, _ = spearmanr(actual, predicted)     
scatter_plot(
    x_ut=actual, 
    y_ut=predicted, 
    path="../outputs/brain_patterner/val-scatter/epoch-"+str(epoch), 
    title="", 
    label=None, 
    dpi=50, 
    scatter_color='#0077b6', 
    title_add=False
)
print(
    f"TEST Loss: {test_loss:.4f}, " 
    f"R2: {r2_test:.4f}",
    f"Spearman: {spearman_corr:.4f}",
) 