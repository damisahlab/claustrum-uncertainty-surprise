import argparse
from argparse import ArgumentParser
import torch
import os 
import time
import multiprocessing



def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)
 

def make_directroy(args, interested_behaviours, make_directroy): 
    root_dir = '../../outputs/' 
    mkdir_fun(root_dir)

    root_dir = '../../outputs/neuropair.net/'
    mkdir_fun(root_dir)

    root_dir = '../../outputs/neuropair.net/repeat-'+make_directroy+'/' 
    mkdir_fun(root_dir)

    root_dir = root_dir + args.subject_id+'/'
    mkdir_fun(root_dir)

    root_dir = root_dir + args.brain_region + '/' 
    mkdir_fun(root_dir)
    
    key_save_dir = root_dir
    mkdir_fun(os.path.join(key_save_dir, 'keys'))
    

    root_dir = os.path.join(root_dir+args.model)
    mkdir_fun(root_dir)
    
    root_dir = root_dir+'/'+ interested_behaviours + '-' + args.output_key
    mkdir_fun(root_dir) 
     
      
    mkdir_fun(os.path.join(root_dir, 'model'))   
    mkdir_fun(os.path.join(root_dir, 'codes')) 
    mkdir_fun(os.path.join(root_dir, 'scatter_plot_validation'))       
    mkdir_fun(os.path.join(root_dir, 'scatter_plot_test'))  
    mkdir_fun(os.path.join(root_dir, 'attention_weights')) 
 
    return root_dir, key_save_dir 
 


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

 
def argparser_fn(subject_id='sub016A', brain_region='ACC', dataset_type=''):
    parser = ArgumentParser(description=f"Set up environment and processing parameters for the {dataset_type} dataset.") 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Dataset specific parameters 
    parser.add_argument('--data_dir', type=str, default='/gpfs/radev/home/aa2793/project/brainnn/datasets/neuropair/'+subject_id+'/', help='Directory containing the data files')
    parser.add_argument('--brain_region', type=str, default=brain_region, help='neural network model')
    parser.add_argument('--subject_id', type=str, default=subject_id, help='neural network model')
 
    parser.add_argument('--noise_reg', type=bool, default=False, help='noise regularization to reduce the effect of overfitting')
    parser.add_argument('--noise_th', type=float, default=0.5, help='Noise Lavel on the training data')

    parser.add_argument('--model', type=str, default='neuropair', help='neural network model')
    parser.add_argument('--autoregressive_model', type=str, default='LSTM', help='neural network model') 
    parser.add_argument('--rnn_num_layers', type=int, default=2, help='RNN num of layers')
    parser.add_argument('--input_size', type=int, default=40, help='RNN num of layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='RNN hidden size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    parser.add_argument('--gat_n_heads', type=int, default=4, help='Number of heads in GAT')
    if brain_region == 'ACC': 
        num_neurons = 21
    elif brain_region == 'CLA': 
        num_neurons=33

    parser.add_argument('--num_neurons', type=int, default=num_neurons, help='Number of Neurons')
    parser.add_argument('--n_behaviour', type=int, default=1, help='Number of Bahaviour') 
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs') 
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-9, help='Weight decay for regularization') 
    parser.add_argument('--device', type=str, default=device, help='Device to use for computation (cuda or cpu)') 

    parser.add_argument('--optim_type', type=str, default='ADAM', help='Optimizer of the model')
    parser.add_argument('--loss_type', type=str, default='MSE', help='Loss of the model')
    parser.add_argument('--output_key', type=str, default=time.strftime("%Y%m%d%H%M%S"), help='')
 
    return parser.parse_args() 

  