import argparse
from argparse import ArgumentParser
import torch
import os 
import time
import multiprocessing



def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)
 
def make_directroy(args):  
    root_dir = '../outputs/'
    mkdir_fun(root_dir)

    root_dir = os.path.join(root_dir+args.model)
    mkdir_fun(root_dir)
    
    root_dir = root_dir+'/'+ args.output_key
    mkdir_fun(root_dir) 
     
    mkdir_fun(os.path.join(root_dir, 'model'))   
    mkdir_fun(os.path.join(root_dir, 'scatter_plot_validation'))    
    return root_dir 
 


def print_gpu_info(output_key):
    os.system('cls' if os.name == 'nt' else 'clear')
    if torch.cuda.is_available():
        print("----------------------------------------------------------------")
        print()
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            props = torch.cuda.get_device_properties(device)
            print(f'              Device {i}:  ')
            print(f'                  Name: {props.name} ')
            print(f'                  Memory: {props.total_memory / 1024 ** 3:.2f} GB') 
            print()
    else:
        print('No GPU available.')

    print(f'              Model')
    print('                  '+output_key) 
    print()
    print("----------------------------------------------------------------") 

 
def argparser_fn(dataset_type, batch_size):
    parser = ArgumentParser(description=f"Set up environment and processing parameters for the {dataset_type} dataset.") 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset specific parameters
    parser.add_argument('--data_dir', type=str, default='../dataset/brainpatterner/claustrum/', help='Directory containing the data files')
    parser.add_argument('--subject', type=str, default='sub016A', help='Subject identifier')
    parser.add_argument('--brain_region', type=str, default='CLA', help='Brain region to process')
    parser.add_argument('--condition', type=str, default='', help='Experimental condition')

    parser.add_argument('--model', type=str, default='recurrentgat', help='neural network model')
    parser.add_argument('--autoregressive_model', type=str, default='LSTM', help='neural network model') 
    parser.add_argument('--rnn_num_layers', type=int, default=2, help='RNN num of layers')
    parser.add_argument('--input_size', type=int, default=40, help='RNN num of layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='RNN hidden size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--noise_th', type=float, default=0.5, help='Noise Lavel on the training data')
    parser.add_argument('--gat_n_heads', type=int, default=4, help='Number of heads in GAT')
    parser.add_argument('--num_neurons', type=int, default=33, help='Number of Neurons')
    parser.add_argument('--n_behaviour', type=int, default=21, help='Number of Bahaviour') 
    parser.add_argument('--num_epochs', type=int, default=5000, help='Number of training epochs') 
    parser.add_argument('--learn_rate', type=float, default=0.002, help='Learning rate for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=3e-2, help='Weight decay for regularization') 
    parser.add_argument('--device', type=str, default=device, help='Device to use for computation (cuda or cpu)') 


    parser.add_argument('--optim_type', type=str, default='ADAM', help='Optimizer of the model')
    parser.add_argument('--loss_type', type=str, default='MSE', help='Loss of the model')
    parser.add_argument('--output_key', type=str, default=time.strftime(dataset_type+"--%Y%m%d-%H%M%S"), help='')
 
    return parser.parse_args() 

  