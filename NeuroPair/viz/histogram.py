import torch
import matplotlib.pyplot as plt

def plot_histograms(y, y_labels):
    num_columns = y.shape[1]  # Number of columns based on the second dimension of y
    fig, axes = plt.subplots(5, 4, figsize=(20, 15))  # Adjust the grid size according to your preference
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
    
    for i in range(num_columns):
        ax = axes[i]
        data = y[:, i].numpy()  # Extract the column data and convert to numpy for plotting
        ax.hist(data, bins=30, alpha=0.75)  # You can adjust the number of bins and transparency
        ax.set_title(y_labels[i])
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
    
    for i in range(num_columns, len(axes)):
        axes[i].set_visible(False)  # Hide unused subplots

    plt.tight_layout()
    plt.savefig('ys_his.png')