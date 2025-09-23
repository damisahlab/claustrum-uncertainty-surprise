from data.adj_matrix import calculate_adjacency_matrix
from einops import rearrange  
import torch  
  
def dataset_splitter(args, x, y, y_label):  
    x = x[:, :, 1:]     
    total_size = x.shape[2]     
    train_indices = torch.arange(0, total_size, 3) 
    test_indices = torch.arange(1, total_size, 3)
    val_indices = torch.arange(2, total_size, 3) 

    x_train = x[:, :, train_indices] 
    x_val = x[:, :, val_indices] 
    x_test = x[:, :, test_indices] 

    seq_length, num_neurons, num_samples = x_train.shape   
    adj_matrix = calculate_adjacency_matrix(rearrange(x_train, 't n s -> n (t s)'))   
    adj_matrix = adj_matrix.to(args.device)

    args.n_behaviour = y.shape[1]
    args.num_neurons = num_neurons
    args.input_size = num_samples   
    y_train = y.unsqueeze(1).repeat(1, x_train.shape[1], 1).float().to(args.device)
    y_val = y.unsqueeze(1).repeat(1, x_val.shape[1], 1).float().to(args.device)
    y_test = y.unsqueeze(1).repeat(1, x_test.shape[1], 1).float().to(args.device)

    x_train = x_train.to(args.device) 
    x_val = x_val.to(args.device) 
    x_test = x_test.to(args.device) 

    return args, (x_train, y_train), (x_val, y_val), (x_test, y_test), adj_matrix