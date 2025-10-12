import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import logging

# some weird error when doing boxplots
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)


custom_rcparams = {
    "text.usetex": True, 
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], 
    "font.size": 9,
    "figure.titlesize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10.5,
    "lines.linewidth": 1,
    "axes.linewidth": 0.5,
    "axes.facecolor": "white",
    "axes.grid": False,
    "lines.markersize": 3,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

defaults_colors = {"QR": "#00202e", "Huber": "#bc5090", "OLS": "#ff8531"}

linestyles = {"QR": '-', "Huber": '--', "OLS": '-.'}

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

        line = ax.plot(subset[x], 
                       subset[y], 
                       marker='o', 
                       linestyle=linestyles[name], 
                       label=name)
        
        color = line[0].get_color()
        
        # error band
        if plot_bands is True:
            ax.fill_between(subset[x], 
                            subset[y] - subset['rmse_sem'],
                            subset[y] + subset['rmse_sem'],
                            alpha=0.2, color=color)

def aggregate_results(results,
                      aggregate_x='rho',
                      aggregate_y='SNR',
                      log_rmse=True, 
                      log_df=False):
    # mk grouped df
    grouped_stats = results.groupby(['name', 'degrees_of_freedom', aggregate_x, aggregate_y]).agg({
        'rmse': ['mean', 'sem']}).reset_index()
    grouped_stats.columns = ['name', 'degrees_of_freedom', aggregate_x, aggregate_y, 'rmse_mean', 'rmse_sem']

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
                    aspect=1.3,
                    aggregate_x='rho',
                    aggregate_y='SNR'):
    
    plt.rcParams.update(custom_rcparams)

    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    grouped_stats = aggregate_results(results, 
                                      log_rmse=log_rmse, 
                                      log_df=log_df,
                                      aggregate_x=aggregate_x,
                                      aggregate_y=aggregate_y)

    for (a, b), data_subset in grouped_stats.groupby([aggregate_x, aggregate_y]):
        fig, ax = plt.subplots(figsize=(height * aspect, height))
        
        plot_with_bands(data=data_subset, x="degrees_of_freedom", y="rmse_mean",
                        plot_bands=se_bands, ax=ax)

        # .title() makes SNR lower case which looks weird
        if aggregate_x == 'SNR':
            title = f"{aggregate_x.replace('_', ' ')}: {a}, {aggregate_y.replace('_', ' ')}: {b}"
        elif aggregate_y == 'SNR':
            title = f"{aggregate_y.replace('_', ' ')}: {b}, {aggregate_x.replace('_', ' ')}: {a}"
        else:
            title = f"{aggregate_y.replace('_', ' ').title()}: {b}, {aggregate_x.replace('_', ' ').title()}: {a}"
            
        ax.set_title(title)
        ax.set_xlabel("Log Degrees of Freedom" if log_df else "Degrees of Freedom")
        ax.set_ylabel("Log RMSE Mean" if log_rmse else "RMSE Mean")
        plt.tight_layout()
        plt.legend(title='Method')
        
        if save_path is not None:
            plt.savefig(f"{save_path}_{aggregate_y}{b}_{aggregate_x}{a}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}_{aggregate_y}{b}_{aggregate_x}{a}.pdf", dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

def plot_boxplot(results, 
                colors=None, 
                save_path=None, 
                log_rmse=True,
                log_x=False, 
                height=1.3, 
                aspect=1.3,
                x='degrees_of_freedom',
                y='rmse',
                n_boxplots=3):
    
    plt.rcParams.update(custom_rcparams)
    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    temp = results.copy()

    if log_x is True:
        temp[x] = np.log10(temp[x])
    if log_rmse is True:
        temp['rmse'] = np.log10(temp['rmse'])
    
    if x == 'SNR':
        xname = f"{x.replace('_', ' ')}"
    else:
        xname = f"{x.replace('_', ' ').title()}"
    
    if n_boxplots is not None:
        if n_boxplots < len(temp[x].unique()):
            # select n_boxplots evenly spaced degrees_of_freedom
            df_values = sorted(temp[x].unique())
            selected_dfs = np.linspace(0, len(df_values) - 1, n_boxplots, dtype=int)
            selected_dfs = [df_values[i] for i in selected_dfs]
            temp = temp[temp[x].isin(selected_dfs)]

    fig, ax = plt.subplots(figsize=(height * aspect, height))
    sns.boxplot(data=temp, x=x, y=y, hue='name', ax=ax)
    
    plt.xlabel(f"Log {xname}" if log_x else f"{xname}")
    plt.ylabel("Log RMSE" if log_rmse else "RMSE")
    plt.tight_layout()
    plt.legend(title='Method')
    
    plt.title(f"Boxplot of " + ("Log RMSE" if log_rmse else "RMSE") + " vs " + (f"Log {xname}" if log_x else f"{xname}"))
    
    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_grid(results, 
               colors=None, 
               save_path=None, 
               log_rmse=True,
               log_df=False, 
               se_bands=True, 
               height=1.3, 
               aspect=1.3,
               aggregate_x='rho',
               aggregate_y='SNR'):
        
    plt.rcParams.update(custom_rcparams)
    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    grouped_stats = aggregate_results(results,
                                      aggregate_x=aggregate_x,
                                      aggregate_y=aggregate_y,
                                      log_rmse=log_rmse, 
                                      log_df=log_df)
    
    g = sns.FacetGrid(grouped_stats, row=aggregate_y, col=aggregate_x, margin_titles=True,
                    sharey=False, sharex=True, height=height, aspect=aspect)

    g.map_dataframe(plot_with_bands, x="degrees_of_freedom", y="rmse_mean",
                        plot_bands=se_bands)

    # add axes labels only in the middle
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    
    g.axes[-1, g.axes.shape[1]//2].set_xlabel("Log Degrees of Freedom" if log_df else "Degrees of Freedom")
    g.axes[g.axes.shape[0]//2, 0].set_ylabel("Log RMSE" if log_rmse else "RMSE")
    
    g.axes[2, 0].set_xticks(list(grouped_stats['degrees_of_freedom'].unique()))
    
    #customize titles and margins
    for ax in range(g.axes.shape[1]):
        # .title() makes SNR lower case which looks weird
        if aggregate_x == 'SNR':
            title = f"{aggregate_x.replace('_', ' ')}: {g.col_names[ax]}"
        else:
            title = f"{aggregate_x.replace('_', ' ').title()}: {g.col_names[ax]}"
        g.axes[0, ax].set_title(title)
    
    for ax in range(g.axes.shape[1]):
        if aggregate_y == 'SNR':
            text = f"{aggregate_y.replace('_', ' ')}: {g.row_names[ax]}"
        else:
            text = f"{aggregate_y.replace('_', ' ').title()}: {g.row_names[ax]}"

        g.axes[ax, -1].texts[0].set_text(text)  
            
    
    # add global title
    if log_df is True:
        g.fig.suptitle("Log RMSE vs Log Degrees of Freedom" if log_rmse 
                    else "RMSE vs Log Degrees of Freedom", y=1.02)
    else:
        g.fig.suptitle("Log RMSE vs Degrees of Freedom" if log_rmse 
                    else "RMSE vs Degrees of Freedom", y=1.02)
        
    g.add_legend()
    
    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
    return g
