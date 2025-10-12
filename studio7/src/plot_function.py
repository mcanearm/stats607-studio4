import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

custom_rcparams = {
    "text.usetex": True, 
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    "font.size": 15,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "lines.linewidth": 2,
    "axes.linewidth": 1,
    "axes.facecolor": "white",    
    "axes.grid": False,
}

defaults_colors = {"QR": "#1f77b4", "Huber": "#ff7f0e", "OLS": "#2ca02c"}
base_path = "studio7/sim_outputs"

def load_data(base_path=None):
    if base_path is None:
        raise ValueError("You must provide a base path to load the data from.")
    
    dfs = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pkl"):
                full_path = os.path.join(root, file)
                dfs.append(pd.read_pickle(full_path))
    
    if not dfs:
        raise ValueError(f"No .pkl files found in the directory: {base_path}")
    
    return pd.concat(dfs, ignore_index=True)

def plot_with_bands(x, y, **kwargs):
    data = kwargs.pop('data')
    plot_bands = kwargs.pop('plot_bands', True)
    ax = plt.gca()
    for name in data['name'].unique():
        subset = data[data['name'] == name].sort_values(x)

        line = ax.plot(subset[x], subset[y], 
                       marker='o', linestyle='--', label=name)
        color = line[0].get_color()
        
        # error band
        ax.fill_between(subset[x], 
                        subset[y] - subset['rmse_sem'],
                        subset[y] + subset['rmse_sem'],
                        alpha=0.2, color=color)

def aggregate_results(results, log_rmse=True, log_df=False):
    # mk grouped df
    grouped_stats = results.groupby(['name', 'degrees_of_freedom', 'rho', 'SNR']).agg({
        'rmse': ['mean', 'sem']}).reset_index()
    grouped_stats.columns = ['name', 'degrees_of_freedom', 'rho', 'SNR', 'rmse_mean', 'rmse_sem']

    if log_rmse is True:
        grouped_stats['rmse_mean'] = np.log10(grouped_stats['rmse_mean'])
        grouped_stats['rmse_sem'] = grouped_stats['rmse_sem'] / grouped_stats['rmse_mean']
    
    if log_df is True:
        grouped_stats['degrees_of_freedom'] = np.log10(grouped_stats['degrees_of_freedom'])
    
    return grouped_stats
    

def plot_individual(results,
                    colors=None, 
                    save_path=None, 
                    log_rmse=True,
                    log_df=False, 
                    se_bands=True, 
                    height=1.3, 
                    aspect=1.3):
    
    plt.rcParams.update(custom_rcparams)

    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    grouped_stats = aggregate_results(results, log_rmse=log_rmse, log_df=log_df)

    for (snr, rho), data_subset in grouped_stats.groupby(['SNR', 'rho']):
        fig, ax = plt.subplots(figsize=(height * aspect, height))
        
        # Call your plotting function directly
        plot_with_bands(data=data_subset, x="degrees_of_freedom", y="rmse_mean",
                        plot_bands=se_bands, ax=ax)

        ax.set_title(f"SNR: {snr}, rho: {rho}")
        ax.set_xlabel("Degrees of Freedom")
        ax.set_ylabel("RMSE Mean")
        plt.tight_layout()
        plt.legend(title='Method')
        
        if save_path is not None:
            plt.savefig(f"{save_path}_SNR{snr}_rho{rho}.png", dpi=300, bbox_inches='tight')
        else:
            plt.show()

def plot_grid(results, 
               colors=None, 
               save_path=None, 
               log_rmse=True,
               log_df=False, 
               se_bands=True, 
               height=1.3, 
               aspect=1.3):
    
    plt.rcParams.update(custom_rcparams)
    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    grouped_stats = aggregate_results(results, log_rmse=log_rmse, log_df=log_df)
    
    # Create FacetGrid
    g = sns.FacetGrid(grouped_stats, row="SNR", col="rho", margin_titles=True,
                    sharey=False, sharex=True, height=height, aspect=aspect)

    g.map_dataframe(plot_with_bands, x="degrees_of_freedom", y="rmse_mean",
                    plot_bands=se_bands)
    
    # add axes labels only in the middle
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
    g.axes[2, 1].set_xlabel("Log Degrees of Freedom" if log_df else "Degrees of Freedom")
    g.axes[1, 0].set_ylabel("Log RMSE" if log_rmse else "RMSE")
    
    g.axes[2, 0].set_xticks(list(grouped_stats['degrees_of_freedom'].unique()))
    
    # add global title
    if log_df is True:
        g.fig.suptitle("Log RMSE vs Log Degrees of Freedom" if log_rmse 
                       else "RMSE vs Log Degrees of Freedom", y=1.02)
    else:
        g.fig.suptitle("Log RMSE vs Degrees of Freedom" if log_rmse 
                       else "RMSE vs Degrees of Freedom", y=1.02)
        
    g.add_legend()
    #plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    return g