import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

colors = {"ols": "#1f77b4", "quantile": "#ff7f0e", "huber": "#2ca02c"}

plt.rcParams.update(custom_rcparams)



def plot_figure(results, colors, path=None, height=3, aspect=1):
    plt.rcParams.update(custom_rcparams)

    if path is None:
        raise ValueError("You must provide a path to save the figure.")
    
    plt.rcParams.update(custom_rcparams)
    
    temp_df = results.copy()
    temp_df['corr'] = temp_df['corr'].apply(lambda x: f"Corr = {x}")
    temp_df['snr'] = temp_df['snr'].apply(lambda x: f"SNR = {x}")

    g = sns.FacetGrid(temp_df, row="snr", col="corr", margin_titles=True,
                    sharey=True, sharex=True, height=height, aspect=aspect)

    g.map_dataframe(sns.lineplot, x="dfs", y="mse", hue="method", style="method",
                    palette=colors, markers=False, dashes=True, legend="full")
    
    g.add_legend(title="Method")

    g.figure.suptitle("Mse vs Degrees of Freedom for different setup", 
                      y=1.02)

    # Adjust layout and titles

    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")

    g.axes[2, 0].set_ylabel("MSE")
    g.axes[2, 0].set_xlabel("Degrees of Freedom")

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.figure.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(path, bbox_inches='tight', dpi=300)