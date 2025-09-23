from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import os

def mkdir_fun(path):
    if not os.path.exists(path): 
        os.mkdir(path)

def scatter_plot(x_ut, y_ut, path, title, label=None, dpi=20, scatter_color='#0077b6', title_add=False): 
    """
    Create a scatter plot of x_ut vs y_ut, compute Spearman correlation and R^2,
    and save the plot. The x- and y-axis ranges are determined by the data's min and max.
    """
    columns = ["GT_Tensor", "PRED_Tensor"] 
    x_ut = x_ut.flatten() 
    y_ut = y_ut.flatten() 
    data_ut = np.column_stack((x_ut, y_ut))
    df_ut = pd.DataFrame(data_ut, columns=columns)

    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(df_ut["GT_Tensor"], df_ut["PRED_Tensor"], 
                         s=5, c=scatter_color, alpha=0.2)

    # Determine axis limits based on data
    x_min, x_max = x_ut.min(), x_ut.max()
    # x_min, x_max = 0, x_ut.max()
    y_min, y_max = y_ut.min(), y_ut.max()

    # Plot the diagonal line y = x from the lowest to the highest across x & y
    lowest_val = min(x_min, y_min) #*0.95
    highest_val = max(x_max, y_max)#*0.95
    line_ideal, = ax.plot([lowest_val, highest_val], 
                          [lowest_val, highest_val], 
                          color='black', linestyle='--', label='Ideal Fit (y = x)')

    # # Fit linear regression and plot the regression line
    # lm = LinearRegression()  
    # lm.fit(x_ut.reshape(-1, 1), y_ut.reshape(-1, 1))
    # x_range = np.linspace(x_min, x_max, 100)
    # y_range = lm.predict(x_range.reshape(-1, 1)) 
    # ax.plot(x_range, y_range, color='red', label='Linear Reg.')

    # Compute Spearman correlation and R² score
    spearman_corr, _ = spearmanr(x_ut, y_ut)   
    r2 = r2_score(x_ut, y_ut) 

    # Round for display
    spearman_corr = round(spearman_corr, 2)
    r2 = round(r2, 2)

    # Custom legend entry for Spearman correlation and R²
    if title_add:
        empty_handle = Line2D(
            [], [], linestyle='none', 
            label= title + f'\n     Spearman Corr.: {spearman_corr}'#      R²: {r2}'
        )

    # Set labels with increased font size
    ax.set_xlabel("Ground Truth", fontsize=15)
    ax.set_ylabel("Predicted Values", fontsize=15)

    # Set tick labels to be thicker
    transparent_color = (0, 0, 0, 0.5)  
    ax.xaxis.set_tick_params(width=10, size=8, labelsize=15, color=transparent_color)
    ax.yaxis.set_tick_params(width=10, size=8, labelsize=15, color=transparent_color)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 

    # Set the actual limits for x and y axes 
    # if x_min<=0: 
    #     ax.set_xlim([+0.05, x_max*.99])
    #     ax.set_ylim([-0.05, y_max*.99])
    # else:
    #     ax.set_xlim([+0.05, x_max*1.])
    #     ax.set_ylim([-0.05, y_max-0.05]) 

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    # Add the custom legend
    if title_add:
        handles = [empty_handle]
        labels = [h.get_label() for h in handles]
        ax.legend(handles=handles, labels=labels, loc='upper left', handlelength=0,
                  handletextpad=0, fontsize=15, frameon=False, 
                  bbox_to_anchor=(0, 1.12))

    plt.tight_layout(pad=0)
    png_path = path
 
    plt.savefig(f"{png_path}.png", dpi=dpi)
    plt.savefig(f"{png_path}.pdf", dpi=dpi)
    plt.close()

    return spearman_corr, r2

def scatter_plotplot(x_ut, y_ut, path, title, label=None, dpi=20, scatter_color='#0077b6', title_add=False): 
    """
    Create a scatter plot of x_ut vs y_ut, compute Spearman correlation and R^2,
    and save the plot. The x- and y-axis ranges are determined by the data's min and max.
    """
    columns = ["GT_Tensor", "PRED_Tensor"] 
    x_ut = x_ut.flatten() 
    y_ut = y_ut.flatten() 
    data_ut = np.column_stack((x_ut, y_ut))
    df_ut = pd.DataFrame(data_ut, columns=columns)

    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(
        df_ut["GT_Tensor"], 
        df_ut["PRED_Tensor"], 
        s=6, 
        facecolors='none',        # No fill inside
        edgecolors='black',   # Light blue edges
        alpha=0.2                 # Optional transparency
    )

    # Determine axis limits based on data
    x_min, x_max = x_ut.min(), x_ut.max()
    # x_min, x_max = 0, x_ut.max()
    y_min, y_max = y_ut.min(), y_ut.max()

    # Plot the diagonal line y = x from the lowest to the highest across x & y
    lowest_val = min(x_min, y_min)*0.95
    highest_val = max(x_max, y_max)*0.95
    line_ideal, = ax.plot([lowest_val, highest_val], 
                          [lowest_val, highest_val], 
                          color='blue', linestyle='--', label='Ideal Fit (y = x)')

    # # Fit linear regression and plot the regression line
    lm = LinearRegression()  
    lm.fit(x_ut.reshape(-1, 1), y_ut.reshape(-1, 1))
    x_range = np.linspace(x_min, x_max, 100)
    y_range = lm.predict(x_range.reshape(-1, 1)) 
    ax.plot(x_range, y_range, color='red', label='Linear Reg.')

    # Compute Spearman correlation and R² score
    spearman_corr, _ = spearmanr(x_ut, y_ut)   
    r2 = r2_score(x_ut, y_ut) 

    # Round for display
    spearman_corr = round(spearman_corr, 2)
    r2 = round(r2, 2)

    # Custom legend entry for Spearman correlation and R²
    if title_add:
        empty_handle = Line2D(
            [], [], linestyle='none', 
            label= title + f'\n     Spearman Corr.: {spearman_corr}'#      R²: {r2}'
        )

    # Set labels with increased font size
    ax.set_xlabel("Ground Truth", fontsize=15)
    ax.set_ylabel("Predicted Values", fontsize=15)

    # Set tick labels to be thicker
    transparent_color = (0, 0, 0, 0.5)  
    ax.xaxis.set_tick_params(width=10, size=8, labelsize=15, color=transparent_color)
    ax.yaxis.set_tick_params(width=10, size=8, labelsize=15, color=transparent_color)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 

    # Set the actual limits for x and y axes 
    # if x_min<=0: 
    #     ax.set_xlim([+0.05, x_max*.99])
    #     ax.set_ylim([-0.05, y_max*.99])
    # else:
    #     ax.set_xlim([+0.05, x_max*1.])
    #     ax.set_ylim([-0.05, y_max-0.05]) 
    ax.set_xlim([-0.05, 1.03])
    ax.set_ylim([-0.05, 1.03])
    # Add the custom legend
    if title_add:
        handles = [empty_handle]
        labels = [h.get_label() for h in handles]
        ax.legend(handles=handles, labels=labels, loc='upper left', handlelength=0,
                  handletextpad=0, fontsize=15, frameon=False, 
                  bbox_to_anchor=(0, 1.12))

    plt.tight_layout(pad=0)
    png_path = path
 
    plt.savefig(f"{png_path}.png", dpi=dpi)
    plt.savefig(f"{png_path}.pdf", dpi=dpi)
    plt.close()

    return spearman_corr, r2