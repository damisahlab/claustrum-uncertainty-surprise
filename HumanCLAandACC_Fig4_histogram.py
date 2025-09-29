"""
Analyze behavioral distributions and y-position data across subjects (ACC/CLA).
Generates percentile-based histograms, KDE plots, and group-level statistics
(kurtosis and permutation tests) for low vs. high behavioral conditions.
This code produces:
    Figure 4c,d
    Extended Data Fig. 8b

IDE: Spyder
Date: 03/2025
"""
#%% ###########################################################################
# Part 1 - Extended Data Fig. 8b, left column
#%% Libraries
from IPython import get_ipython
get_ipython().magic('reset -sf')  # Reset environment
get_ipython().magic('clear')

import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

#%% Parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 5
plt.rcParams["axes.linewidth"] = 0.5

# Subjects
behaviors = {'sub016A', 'sub016B', 'sub017', 'sub019A', 'sub019B', 'sub020', 'sub023', 'sub024A'}
low_percentile = 0.3  
high_percentile = 1 - low_percentile

data_path = rf'define data path'
save_path = rf'define save path'
os.makedirs(save_path, exist_ok=True)  # Ensure output directory exists

#%% Loop Through Subjects
for sub in behaviors:
    file_path = rf'{data_path}/behaviors/{sub}.xlsx'
    
    # Check if file exists before proceeding
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Load behavior data
    behavior = pd.read_excel(file_path)
    columns_to_keep = ["Subject", "trial", "outcome", "A_safety_variance", "B_safety_variance"]
    behavior = behavior[columns_to_keep]

    #%% Plot histograms & KDE for each behavior metric
    for behavior_name in behavior.columns[3:]:  # Skip Subject, trial, outcome
        data = behavior[behavior_name].dropna()  # Remove NaN values
        
        if data.empty:
            print(f"No data for {sub}-{behavior_name}")
            continue

        low_threshold = data.quantile(low_percentile)
        high_threshold = data.quantile(high_percentile)

        # KDE estimation
        kde = gaussian_kde(data, bw_method=0.15)
        x = np.linspace(min(data), max(data), 1000)
        y = kde(x)

        # Create plot
        fig, ax = plt.subplots(figsize=(1.5, 1))
        ax.hist(data, bins=20, alpha=0.7, color='gray', edgecolor='black', linewidth=0.3, density=True)
        ax.plot(x, y, color='red', linewidth=0.5, label='KDE')
        ax.axvline(low_threshold, color='blue', linestyle='dashed', linewidth=0.5, label=f'P30 = {low_threshold:.4f}')
        ax.axvline(high_threshold, color='orange', linestyle='dashed', linewidth=0.5, label=f'P70 = {high_threshold:.4f}')
        
        ax.set_xlabel(behavior_name, labelpad=1)
        ax.set_ylabel("Density", labelpad=1)
        # ax.set_title(f"{brain_region}", pad=2)
        ax.locator_params(axis="x", nbins=3)
        ax.locator_params(axis="y", nbins=3)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()

        # Adjust plot aesthetics
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)    
        ax.tick_params(axis='both', which='both', length=1, width=0.5, pad=1)

        # Save plot
        save_path_svg = os.path.join(save_path, f"{sub}-{behavior_name}.svg")
        save_path_pdf = os.path.join(save_path, f"{sub}-{behavior_name}.pdf")
        plt.savefig(save_path_svg, dpi=300, bbox_inches="tight")
        plt.savefig(save_path_pdf, dpi=300, bbox_inches="tight")
        plt.close()

#%% ###########################################################################
# Part 2 - Extended Data Fig. 8b, right column and Fig. 4c,d
#%% Libraries
from IPython import get_ipython
get_ipython().magic('reset -sf')  # Reset environment, '-sf' forces a reset without confirmation
get_ipython().magic('clear')

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import gaussian_kde, kurtosis, wilcoxon
import scipy.stats as stats
from itertools import combinations

#%% Functions
# Plot A conditions
def dot_histogram_yaxis(ax, data, bins, color, offset=0, dot_size=.1, max_dots=50):
    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize dot count to make visualization proportional
    counts_scaled = counts / counts.max() * max_dots

    for i, (center, scaled) in enumerate(zip(bin_centers, counts_scaled)):
        for j in range(int(scaled)):
            x = j * 0.01 + offset  # spread horizontally
            y = center + np.random.uniform(-6, 6)
            ax.plot(x, y, 'o', color=color, markersize=dot_size, alpha=0.5)

def permutationTest(sample1, sample2, permutations, sidedness, exact=False):
    """
    permutation_test adapted from :
    Laurens R Krol (2025). Permutation Test (https://github.com/lrkrol/permutationTest), GitHub. Retrieved March 20, 2025.    
    Created on Thu Mar 20 13:11:51 2025
    @author: rd883
    """        
    p, observed_difference, effect_size = 10, 10, 10
    
    sample1, sample2 = np.asarray(sample1), np.asarray(sample2)
    observed_difference = np.nanmean(sample1) - np.nanmean(sample2)
    pooled_std = np.sqrt(((len(sample1) - 1) * np.nanvar(sample1, ddof=1) + 
                          (len(sample2) - 1) * np.nanvar(sample2, ddof=1)) /
                         (len(sample1) + len(sample2) - 2))
    # effect_size = observed_difference / pooled_std
    
    all_observations = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    
    if exact:
        all_combinations = list(combinations(range(len(all_observations)), n1))
        permutations = len(all_combinations)
    
    random_differences = np.zeros(permutations)
    
    for i in range(permutations):
        if exact:
            indices = np.array(all_combinations[i])
        else:
            np.random.shuffle(all_observations)
            indices = np.arange(n1)
        
        perm_sample1 = all_observations[indices]        
        perm_sample2 = np.delete(all_observations, indices)        
        random_differences[i] = np.nanmean(perm_sample1) - np.nanmean(perm_sample2)
    
    if sidedness == 'both':
        p = (np.sum(np.abs(random_differences) > abs(observed_difference)) + 1) / (permutations + 1)
    elif sidedness == 'smaller':
        p = (np.sum(random_differences < observed_difference) + 1) / (permutations + 1)
    elif sidedness == 'larger':
        p = (np.sum(random_differences > observed_difference) + 1) / (permutations + 1)
    else:
        raise ValueError("sidedness must be 'both', 'smaller', or 'larger'")
       
    return p, observed_difference#, effect_size
                  
#%% Main execution
# Set global font properties for plots
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 5
plt.rcParams["axes.linewidth"] = 0.5

data_path = rf'define data path'
save_path = rf'define save path'
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

# Identify unique subject IDs and conditions
subjects = ["016A", "016B", "017", "019A", "019B", "020", "023", "024A"]
conditions = ["A_High", "A_Low", "B_High", "B_Low"]
idx = 0
kurt_A_high, kurt_A_low, kurt_B_high, kurt_B_low = [], [], [], []
# Loop through subjects
for subject in subjects:
    data_dict = {}
    
    # Load data for each condition
    for condition in conditions:
        filename = f"Uncertainty_Percentile_30-70_4_sec_PositionChange_{condition}_Uncertainty_distribution_Sub{subject}.mat"
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            data = scipy.io.loadmat(filepath)['data_no_zeros'].flatten()
            data_dict[condition] = data
    
    # Check if all required data is available
    if "A_High" in data_dict and "A_Low" in data_dict and "B_High" in data_dict and "B_Low" in data_dict:
        # Stat on histogram
        # Kurtosis measures the "tailedness" or concentration of data around the mean. 
        # A higher kurtosis value indicates that the data is more concentrated around 
        # certain values (i.e., a preference for certain positions), while a lower 
        # kurtosis suggests the data is more spread out (i.e., movement across more positions).
        num_bins = 49  # Adjust as needed
        A_hist_high, _ = np.histogram(data_dict["A_High"], bins=num_bins, density=False)
        A_hist_low, _ = np.histogram(data_dict["A_Low"], bins=num_bins, density=False)
        B_hist_high, _ = np.histogram(data_dict["B_High"], bins=num_bins, density=False)
        B_hist_low, _ = np.histogram(data_dict["B_Low"], bins=num_bins, density=False)        
        kurt_A_high.append(kurtosis(A_hist_high))  # Compute kurtosis and append
        kurt_A_low.append(kurtosis(A_hist_low))
        kurt_B_high.append(kurtosis(B_hist_high))
        kurt_B_low.append(kurtosis(B_hist_low))
        
        # KDE estimation
        kde_A_High = gaussian_kde(data_dict["A_High"], bw_method=0.15)
        kde_A_Low = gaussian_kde(data_dict["A_Low"], bw_method=0.15)
        kde_B_High = gaussian_kde(data_dict["B_High"], bw_method=0.15)
        kde_B_Low = gaussian_kde(data_dict["B_Low"], bw_method=0.15)
        A_x_high = np.linspace(min(data_dict["A_High"]), max(data_dict["A_High"]), 1000)
        A_x_low = np.linspace(min(data_dict["A_Low"]), max(data_dict["A_Low"]), 1000)
        B_x_high = np.linspace(min(data_dict["B_High"]), max(data_dict["B_High"]), 1000)
        B_x_low = np.linspace(min(data_dict["B_Low"]), max(data_dict["B_Low"]), 1000)
        A_y_high = kde_A_High(A_x_high)
        A_y_low = kde_A_Low(A_x_low)
        B_y_high = kde_B_High(B_x_high)
        B_y_low = kde_B_Low(B_x_low)
            
        # Create plot
        fig, ax = plt.subplots(1, 2, figsize=(3.25, 0.75))
        
        # Plot A conditions
        ax[0].hist(data_dict['A_Low'], bins=num_bins, alpha=0.5, color='blue', edgecolor='black', linewidth=0.3, density=True)
        ax[0].hist(data_dict['A_High'], bins=num_bins, alpha=0.5, color='orange', edgecolor='black', linewidth=0.3, density=True)                
        ax[0].plot(A_x_high, A_y_high, color='orange', linewidth=0.5)
        ax[0].plot(A_x_low, A_y_low, color='blue', linewidth=0.5)
        ax[0].set_xlabel('y-position', labelpad=1)
        ax[0].set_ylabel("density", labelpad=1)        
        ax[0].locator_params(axis="x", nbins=3)
        ax[0].locator_params(axis="y", nbins=3)
        ax[0].set_xticks([0, 300, 600])
        ax[0].locator_params(axis="y", nbins=3)
        
        # Plot B conditions
        ax[1].hist(data_dict['B_Low'], bins=num_bins, alpha=0.5, color='blue', edgecolor='black', linewidth=0.3, density=True)
        ax[1].hist(data_dict['B_High'], bins=num_bins, alpha=0.5, color='orange', edgecolor='black', linewidth=0.3, density=True)        
        ax[1].plot(B_x_high, B_y_high, color='orange', linewidth=0.5)
        ax[1].plot(B_x_low, B_y_low, color='blue', linewidth=0.5)
        ax[1].set_xlabel('y-position', labelpad=1)
        ax[1].set_ylabel("density", labelpad=1)        
        ax[1].locator_params(axis="x", nbins=3)
        ax[1].locator_params(axis="y", nbins=3)   
        ax[1].set_xticks([0, 300, 600])
        ax[1].locator_params(axis="y", nbins=3)
        
        # Adjust plot aesthetics
        for i in range(2):
            for spine in ax[i].spines.values():
                spine.set_linewidth(0.25)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].tick_params(axis='both', which='both', length=1, width=0.5, pad=1)
        
        # Save the plot
        plt.savefig(os.path.join(save_path, f"Uncertainty_Distribution_Sub{subject}.svg"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(save_path, f"Uncertainty_Distribution_Sub{subject}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()

# only the significants from A or B
new_high = []
new_low = []
# Define the custom index assignment
a_idx = [0, 3, 6, 7]  # Indices from A
b_idx = [1, 2, 4, 5]      # Indices from B

# Fill in new high and low lists following the pattern
for i in range(len(kurt_A_high)):  # Assuming same length for A and B
    if i in a_idx:
        new_high.append(kurt_A_high[i])
        new_low.append(kurt_A_low[i])
    elif i in b_idx:
        new_high.append(kurt_B_high[i])
        new_low.append(kurt_B_low[i])

_, p_value = wilcoxon(new_high, new_low)

#%% Plot Stats
# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(.975, .85))  # Adjust figure size as needed
# Scatter plot for individual data points
# Calculate means and SEMs
mean_low = np.mean(new_low)
sem_low = stats.sem(new_low)
mean_high = np.mean(new_high)
sem_high = stats.sem(new_high)
ax.scatter(np.ones(len(new_low)), new_low, 5, facecolors='none', edgecolors='blue', marker='^', label='Low', linewidths=0.25)
ax.scatter(2*np.ones(len(new_high)), new_high, 5, facecolors='none', edgecolors='orange', marker='^', label='High', linewidths=0.25)
    
# Scatter plot for means
ax.scatter(1, np.mean(new_low), 6, 'k', marker='^', label='Mean Low', edgecolors='k', linewidths=0.2)
ax.scatter(2, np.mean(new_high), 6, 'k', marker='v', label='Mean High', edgecolors='k', linewidths=0.2)

# Formatting axes
ax.set_xlim([0, 3])
ax.set_xticks([1, 2])
ax.set_xticklabels(['Low', 'High'])

# Set y-limits dynamically
yl = [min(np.min(new_low), np.min(new_high)), max(np.max(new_low), np.max(new_high))]
m = yl[1] - yl[0]
ax.set_ylim([yl[0] - 0.05 * m, yl[1] + m * 0.25])
ax.set_ylabel(r'kurtosis')

# Connect paired data points
for i in range(len(new_low)):
    ax.plot([1, 2], [new_low[i], new_high[i]], '-k', linewidth=0.075)

# Line connecting means
ax.plot([1, 2], [np.mean(new_low), np.mean(new_high)], '-k', linewidth=0.2)

# Asterisk or "ns" for significance
asterisk_y = yl[1] + m * 0.125
bar_y = asterisk_y + m * 0.07

# Format the annotation text
mean_sem_text = (
    f"{'**' if p_value < 0.05 else 'ns'} (p={p_value:.3f})\n"
    f"Low: {mean_low:.2f}±{sem_low:.2f}\n "
    f"High: {mean_high:.2f}±{sem_high:.2f}"
)

if p_value < .05:
    ax.text(1.5, bar_y, mean_sem_text, ha='center')
else:
    ax.text(1.5, bar_y, mean_sem_text, ha='center')

# Style adjustments
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(axis='both', which='both', length=1, width=0.5, pad=1)
ax.set_xlabel(ax.get_xlabel(), labelpad=1)
ax.set_ylabel(ax.get_ylabel(), labelpad=1)

# Show plot
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(save_path, "kurtosis.svg"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(save_path, "kurtosis.pdf"), dpi=300, bbox_inches="tight")
plt.close()