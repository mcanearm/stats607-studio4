import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_figure(results, colors):
    colorols = colors['ols']
    colorquantile = colors['quantile']
    colorhuber = colors['huber']
    
    res_ols = results[results['method'] == 'ols']
    res_quantile = results[results['method'] == 'quantile']
    res_huber = results[results['method'] == 'huber']
    
    fig, ax = plt.subplots(3, 3, figsize=(10,10))

    ax[0, 0].set_xlabel('Degrees of Freedom')
    ax[0, 0].set_ylabel('MSE')

    snrs = results['snr'].unique()
    corrs = results['corr'].unique()

    for i, snr in enumerate(snrs):
        for j, corr in enumerate(corrs):
            ax[i, j].set_title(f'SNR: {snr}, Corr: {corr}')

    for x in range(3):
        for y in range(3):
            ax[x, y].plot(res_ols['dfs'], res_ols['mse'], color=colorols, label='OLS')
            ax[x, y].plot(res_quantile['dfs'], res_quantile['mse'], color=colorquantile, label='Quantile')
            ax[x, y].plot(res_huber['dfs'], res_huber['mse'], color=colorhuber, label='Huber')

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize='medium')
    plt.suptitle('MSE vs Degrees of Freedom for Different setups', fontsize=16)
    plt.tight_layout()