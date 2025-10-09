import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

custom_rcparams = {
    "text.usetex": True,  # render all text with LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # now this works via LaTeX

    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    "lines.linewidth": 1.5,
    "axes.linewidth": 0.5,
}

def plot_figure(results, colors, path=None):
    if path is None:
        raise ValueError("You must provide a path to save the figure.")
    
    plt.rcParams.update(custom_rcparams)
    
    colorols = colors['ols']
    colorquantile = colors['quantile']
    colorhuber = colors['huber']
    
    res_ols = results[results['method'] == 'ols']
    res_quantile = results[results['method'] == 'quantile']
    res_huber = results[results['method'] == 'huber']
    
    fig, ax = plt.subplots(3, 3, figsize=(10,10))

    ax[0, 2].set_xlabel('Degrees of Freedom')
    ax[0, 2].set_ylabel('MSE')

    snrs = results['snr'].unique()
    corrs = results['corr'].unique()

    # plot each (snr, corr) pair in the corresponding subplot and label rows/columns
    for i, snr in enumerate(snrs):
        for j, corr in enumerate(corrs):
            sel_ols = res_ols[(res_ols['snr'] == snr) & (res_ols['corr'] == corr)]
            sel_quant = res_quantile[(res_quantile['snr'] == snr) & (res_quantile['corr'] == corr)]
            sel_hub = res_huber[(res_huber['snr'] == snr) & (res_huber['corr'] == corr)]

            ax[i, j].plot(sel_ols['dfs'], sel_ols['mse'], color=colorols, label='OLS')
            ax[i, j].plot(sel_quant['dfs'], sel_quant['mse'], color=colorquantile, label='Quantile')
            ax[i, j].plot(sel_hub['dfs'], sel_hub['mse'], color=colorhuber, label='Huber')

            # label the rows with SNR and the columns with Corr
            if j == 0:
                ax[i, j].set_ylabel(f'SNR = {snr}', rotation=90, labelpad=10)
            if i == len(snrs) - 1:
                ax[i, j].set_xlabel(f'Corr = {corr}')
            #ax[i, j].set_title(f'SNR: {snr}, Corr: {corr}')


    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize='medium')
    plt.suptitle('MSE vs Degrees of Freedom for Different setups', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(path, dpi=300, bbox_inches='tight')