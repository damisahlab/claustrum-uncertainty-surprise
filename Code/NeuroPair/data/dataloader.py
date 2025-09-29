import torch 
import os
import glob
from einops import rearrange
from scipy.io import loadmat
import pandas as pd 
import torch.nn.functional as F 
 

def load_firing_rates(file_path):
    data = loadmat(file_path)
    for key in ['fr', 'hit', 'miss']:
        if key in data:
            return data[key]
    print(f"None of the keys ('fr', 'hit', 'miss') are available in {file_path}")
    return None

def load_behavior_data(file_path):
    return pd.read_excel(file_path)

def process_data(data_dir, subject, brain_region, neuron_label=False): 

    neuron_type = 'cell_types.csv'
    neuron_type_df = pd.read_csv(neuron_type)
 
    data_dir = os.path.join(data_dir, brain_region)  
    neuron_mat_files = glob.glob(os.path.join(data_dir+'/', '*.mat'))    
     
    behavior_data_path = glob.glob(os.path.join(data_dir, brain_region+f'_{subject}.xlsx'))[0]    
    behavior_data = load_behavior_data(behavior_data_path)
    
    behavior_data[behavior_data.columns[1:]] = behavior_data[behavior_data.columns[1:]].astype(float)
    
    numeric_cols = behavior_data.select_dtypes(include=[float]).columns
    numeric_data_matrix = behavior_data[numeric_cols].values 

    data = []
    neuron_type = []
    neuron_file_name = []
    for i, neuron_i_path in enumerate(neuron_mat_files):   
        neuron_i_data = load_firing_rates(neuron_i_path) 
        neuron_i_data = torch.tensor(neuron_i_data).unsqueeze(0)
        data.append(neuron_i_data)   
        neuron_id = neuron_i_path.replace(data_dir, "")[1:-4]

        neuron_file_name.append(neuron_id[-5:])

        brain_region = neuron_id[:3] 
        neuron_type_df_i = neuron_type_df[neuron_type_df['Brain_Region']==brain_region]
        neuron_type_df_i = neuron_type_df_i[neuron_type_df_i['Neuron_ID']==neuron_id[4:]]  
        neuron_type.append(neuron_type_df_i['Cell_Type'].item())
 
    data = torch.cat(data) 
    data = rearrange(data, 'neuron trail signal -> trail neuron signal')   
    
    if neuron_label:
        return data.float(), torch.tensor(numeric_data_matrix), numeric_cols, neuron_type, neuron_file_name
    else:
        return data.float(), torch.tensor(numeric_data_matrix), numeric_cols