import torch
import matplotlib.pyplot as plt
from einops import rearrange  

def singal_plot(x):  
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