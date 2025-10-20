import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress matplotlib category warning for boxplots
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

# Custom plot rcParams
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

# Default color mappings for methods
defaults_colors = {"QR": "#00202e", "Huber": "#bc5090", "OLS": "#ff8531"}

# Default linestyles for methods
linestyles = {"QR": "-", "Huber": "--", "OLS": "-."}

base_path = "studio7/sim_outputs"


def load_data(base_path=None):
    """Load all simulation result .pkl files from the given directory into a DataFrame."""
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
    """
    Plot lines with confidence/error bands for each method.
    """
    data = kwargs.pop("data")
    plot_bands = kwargs.pop("plot_bands", True)
    ax = plt.gca()
    for name in data["name"].unique():
        subset = data[data["name"] == name].sort_values(x)
        line = ax.plot(
            subset[x], subset[y], marker="o", linestyle=linestyles[name], label=name
        )
        color = line[0].get_color()
        # Draw error band if requested
        if plot_bands is True:
            ax.fill_between(
                subset[x],
                subset[y] - subset["rmse_sem"],
                subset[y] + subset["rmse_sem"],
                alpha=0.2,
                color=color,
            )


def aggregate_results(
    results, aggregate_x="rho", aggregate_y="SNR", log_rmse=True, log_df=False
):
    """
    Aggregate results by taking the mean and SEM of RMSE for each method and degrees_of_freedom.
    Optionally apply log10 transform to RMSE values and/or degrees_of_freedom.
    """
    grouped_stats = (
        results.groupby(["name", "degrees_of_freedom", aggregate_x, aggregate_y])
        .agg({"rmse": ["mean", "sem"]})
        .reset_index()
    )
    grouped_stats.columns = [
        "name",
        "degrees_of_freedom",
        aggregate_x,
        aggregate_y,
        "rmse_mean",
        "rmse_sem",
    ]

    if log_rmse is True:
        grouped_stats["rmse_mean"] = np.log10(grouped_stats["rmse_mean"])
        grouped_stats["rmse_sem"] = (
            grouped_stats["rmse_sem"] / grouped_stats["rmse_mean"]
        )

    if log_df is True:
        grouped_stats["degrees_of_freedom"] = np.log10(
            grouped_stats["degrees_of_freedom"]
        )

    return grouped_stats


def plot_individual(
    results,
    colors=None,
    save_path=None,
    log_rmse=True,
    log_df=False,
    se_bands=True,
    height=1.3,
    aspect=1.3,
    aggregate_x="rho",
    aggregate_y="SNR",
):
    """
    Plot individual lineplots for each combination of aggregate_x and aggregate_y.
    """
    plt.rcParams.update(custom_rcparams)

    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    grouped_stats = aggregate_results(
        results,
        log_rmse=log_rmse,
        log_df=log_df,
        aggregate_x=aggregate_x,
        aggregate_y=aggregate_y,
    )

    for (a, b), data_subset in grouped_stats.groupby([aggregate_x, aggregate_y]):
        fig, ax = plt.subplots(figsize=(height * aspect, height))

        plot_with_bands(
            data=data_subset,
            x="degrees_of_freedom",
            y="rmse_mean",
            plot_bands=se_bands,
            ax=ax,
        )

        # Custom facet titles to preserve SNR capitalization
        if aggregate_x == "SNR":
            title = f"{aggregate_x.replace('_', ' ')}: {a}, {aggregate_y.replace('_', ' ')}: {b}"
        elif aggregate_y == "SNR":
            title = f"{aggregate_y.replace('_', ' ')}: {b}, {aggregate_x.replace('_', ' ')}: {a}"
        else:
            title = f"{aggregate_y.replace('_', ' ').title()}: {b}, {aggregate_x.replace('_', ' ').title()}: {a}"

        ax.set_title(title)
        ax.set_xlabel("Log Degrees of Freedom" if log_df else "Degrees of Freedom")
        ax.set_ylabel("Log RMSE Mean" if log_rmse else "RMSE Mean")
        plt.tight_layout()
        plt.legend(title="Method")

        if save_path is not None:
            plt.savefig(
                f"{save_path}_{aggregate_y}{b}_{aggregate_x}{a}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                f"{save_path}_{aggregate_y}{b}_{aggregate_x}{a}.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()


def plot_boxplot(
    results,
    colors=None,
    save_path=None,
    log_rmse=True,
    log_x=False,
    height=1.3,
    aspect=1.3,
    x="degrees_of_freedom",
    y="rmse",
    n_boxplots=3,
):
    """
    Plot RMSE boxplot versus the given x variable. Optionally log-transform RMSE or x,
    and limit number of boxplots.
    """
    plt.rcParams.update(custom_rcparams)
    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    temp = results.copy()

    if log_x is True:
        temp[x] = np.log10(temp[x])
    if log_rmse is True:
        temp["rmse"] = np.log10(temp["rmse"])

    if x == "SNR":
        xname = f"{x.replace('_', ' ')}"
    else:
        xname = f"{x.replace('_', ' ').title()}"

    if n_boxplots is not None:
        if n_boxplots < len(temp[x].unique()):
            # Select n_boxplots evenly spaced along x
            df_values = sorted(temp[x].unique())
            selected_dfs = np.linspace(0, len(df_values) - 1, n_boxplots, dtype=int)
            selected_dfs = [df_values[i] for i in selected_dfs]
            temp = temp[temp[x].isin(selected_dfs)]

    fig, ax = plt.subplots(figsize=(height * aspect, height))
    sns.boxplot(data=temp, x=x, y=y, hue="name", ax=ax)

    plt.xlabel(f"Log {xname}" if log_x else f"{xname}")
    plt.ylabel("Log RMSE" if log_rmse else "RMSE")
    plt.tight_layout()
    plt.legend(title="Method")

    plt.title(
        "Boxplot of "
        + ("Log RMSE" if log_rmse else "RMSE")
        + " vs "
        + (f"Log {xname}" if log_x else f"{xname}")
    )

    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_grid(
    results,
    colors=None,
    save_path=None,
    log_rmse=True,
    log_df=False,
    se_bands=True,
    height=1.3,
    aspect=1.3,
    aggregate_x="rho",
    aggregate_y="SNR",
):
    """
    Plot a grid of RMSE lineplots faceted by aggregate_x and aggregate_y.
    """
    plt.rcParams.update(custom_rcparams)
    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    grouped_stats = aggregate_results(
        results,
        aggregate_x=aggregate_x,
        aggregate_y=aggregate_y,
        log_rmse=log_rmse,
        log_df=log_df,
    )

    g = sns.FacetGrid(
        grouped_stats,
        row=aggregate_y,
        col=aggregate_x,
        margin_titles=True,
        sharey=False,
        sharex=True,
        height=height,
        aspect=aspect,
    )

    g.map_dataframe(
        plot_with_bands, x="degrees_of_freedom", y="rmse_mean", plot_bands=se_bands
    )

    # Remove default x/y axis labels and tick labels from all subplots
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    # Set x and y axis labels only in central places
    g.axes[-1, g.axes.shape[1] // 2].set_xlabel(
        "Log Degrees of Freedom" if log_df else "Degrees of Freedom"
    )
    g.axes[g.axes.shape[0] // 2, 0].set_ylabel("Log RMSE" if log_rmse else "RMSE")

    g.axes[2, 0].set_xticks(list(grouped_stats["degrees_of_freedom"].unique()))

    # Set column facet titles (preserve capitalization for SNR)
    for ax in range(g.axes.shape[1]):
        if aggregate_x == "SNR":
            title = f"{aggregate_x.replace('_', ' ')}: {g.col_names[ax]}"
        else:
            title = f"{aggregate_x.replace('_', ' ').title()}: {g.col_names[ax]}"
        g.axes[0, ax].set_title(title)

    # Set custom row facet labels with preserved SNR capitalization
    for ax in range(g.axes.shape[1]):
        if aggregate_y == "SNR":
            text = f"{aggregate_y.replace('_', ' ')}: {g.row_names[ax]}"
        else:
            text = f"{aggregate_y.replace('_', ' ').title()}: {g.row_names[ax]}"

        g.axes[ax, -1].texts[0].set_text(text)

    # Set figure title at the top
    if log_df is True:
        g.fig.suptitle(
            "Log RMSE vs Log Degrees of Freedom"
            if log_rmse
            else "RMSE vs Log Degrees of Freedom",
            y=1.02,
        )
    else:
        g.fig.suptitle(
            "Log RMSE vs Degrees of Freedom"
            if log_rmse
            else "RMSE vs Degrees of Freedom",
            y=1.02,
        )

    g.add_legend()

    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    return g


# Ensure the dataframe contains coverage and interval width columns; add them if missing


def _agg_sem(x):
    """
    Helper function: compute standard error of the mean for a numeric array x.
    """
    x = np.asarray(x, float)
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _ensure_cov_width_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the DataFrame includes 'coverage_mean' and 'width_mean' columns.
    If necessary, computes them based on 'ci_lower', 'ci_upper', and true beta.
    Returns a new DataFrame (copy) if computation is required; otherwise returns original.
    """
    if "coverage_mean" in df.columns and "width_mean" in df.columns:
        return df

    out = df.copy()
    covs, widths = [], []

    for _, r in out.iterrows():
        lo = np.atleast_1d(r["ci_lower"])
        hi = np.atleast_1d(r["ci_upper"])

        # Do NOT use "A or B" for arrays; choose proper key
        tb_val = r.get("true_beta", None)
        if tb_val is None:
            tb_val = r.get("beta_true", None)

        if tb_val is None:
            # Coverage is NaN if true value is absent; still compute width from CI
            covs.append(np.nan)
            widths.append(float(np.mean(hi - lo)))
            continue

        tb = np.atleast_1d(tb_val)

        covs.append(float(np.mean((tb >= lo) & (tb <= hi))))
        widths.append(float(np.mean(hi - lo)))

    out["coverage_mean"] = covs
    out["width_mean"] = widths
    return out


def plot_coverage(
    results: pd.DataFrame,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    log_df: bool = False,
    ci_level: float = 0.95,
    se_bands: bool = True,
    height: float = 1.3,
    aspect: float = 1.3,
    aggregate_x: str = "rho",
    aggregate_y: str = "SNR",
):
    """
    Plot the coverage of the true beta value versus degrees of freedom, faceted by aggregate_x and aggregate_y.
    Coverage is calculated from existing 'coverage_mean' in the dataframe, or computed from ci_lower/ci_upper.
    """
    plt.rcParams.update(custom_rcparams)
    colors = colors or defaults_colors
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tmp = _ensure_cov_width_columns(results)
    group_cols = ["name", "degrees_of_freedom", aggregate_x, aggregate_y]
    if not set(group_cols).issubset(tmp.columns):
        missing = sorted(set(group_cols) - set(tmp.columns))
        raise ValueError(f"Missing columns for coverage: {missing}")

    stat = (
        tmp.groupby(group_cols)["coverage_mean"]
        .agg(mean="mean", sem=_agg_sem)
        .reset_index()
    )

    if log_df:
        stat["degrees_of_freedom"] = np.log10(stat["degrees_of_freedom"])

    # FacetGrid: turn off margin_titles for right margin column text
    g = sns.FacetGrid(
        stat,
        row=aggregate_y,
        col=aggregate_x,
        margin_titles=False,
        sharey=True,
        sharex=True,
        height=height,
        aspect=aspect,
    )

    def draw(data, **kw):
        ax = plt.gca()
        for m in data["name"].unique():
            d = data[data["name"] == m].sort_values("degrees_of_freedom")
            if d.empty:
                continue
            line = ax.plot(
                d["degrees_of_freedom"],
                d["mean"],
                marker="o",
                linestyle=linestyles.get(m, "-"),
                color=colors.get(m),
                label=m,
            )
            if se_bands:
                ax.fill_between(
                    d["degrees_of_freedom"],
                    d["mean"] - d["sem"],
                    d["mean"] + d["sem"],
                    alpha=0.2,
                    color=line[0].get_color(),
                )
        # Reference line for nominal CI level
        ax.axhline(ci_level, ls="--", lw=1, color="red", alpha=0.7)
        ax.set_ylim(0.0, 1.05)

    g.map_dataframe(draw)

    # Remove all x/y axis labels and subplot titles from subplots
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    # Set unified axis labels (bottom center, left center only)
    label_fs = 12
    g.axes[-1, g.axes.shape[1] // 2].set_xlabel(
        "Log Degrees of Freedom" if log_df else "Degrees of Freedom", fontsize=label_fs
    )
    g.axes[g.axes.shape[0] // 2, 0].set_ylabel("Coverage", fontsize=label_fs)

    # Set column titles for each facet column at the top
    for j in range(g.axes.shape[1]):
        title = f"{aggregate_x if aggregate_x == 'SNR' else aggregate_x.replace('_', ' ').title()}: {g.col_names[j]}"
        g.axes[0, j].set_title(title)

    # Set custom row facet labels (right side, vertical text)
    for i in range(g.axes.shape[0]):
        txt = f"{aggregate_y if aggregate_y == 'SNR' else aggregate_y.replace('_', ' ').title()}: {g.row_names[i]}"
        g.axes[i, -1].text(
            1.02,
            0.95,
            txt,
            transform=g.axes[i, -1].transAxes,
            ha="left",
            va="top",
            rotation=90,
        )

    # Figure title, including percentage for CI target
    sup = (
        "Coverage vs Log Degrees of Freedom"
        if log_df
        else "Coverage vs Degrees of Freedom"
    )
    g.fig.suptitle(sup, y=1.02)

    # Add margin on the right for legend
    g.fig.subplots_adjust(right=0.95)

    # Place legend outside plot area, right margin
    g.add_legend(
        title="Method",
        bbox_to_anchor=(0.85, 0.5),  # Canvas coordinate: just right of grid
        loc="center left",
        frameon=False,
        ncol=1,  # Single column, vertical layout
    )

    # Save to file if path is specified, else show interactively
    if save_path:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return g


def plot_interval_width(
    results: pd.DataFrame,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    log_df: bool = False,
    log_width: bool = False,
    se_bands: bool = True,
    height: float = 1.3,
    aspect: float = 1.3,
    aggregate_x: str = "rho",
    aggregate_y: str = "SNR",
):
    """
    Plot the average confidence interval width as a function of degrees of freedom, faceted by aggregate_x and aggregate_y.
    Uses width_mean if present, otherwise computes from ci_upper - ci_lower.
    Supports log transform of both DoF and CI width.
    """
    plt.rcParams.update(custom_rcparams)
    colors = colors or defaults_colors
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Ensure width_mean present in dataframe; add if missing
    tmp = _ensure_cov_width_columns(results)
    group_cols = ["name", "degrees_of_freedom", aggregate_x, aggregate_y]
    if not set(group_cols).issubset(tmp.columns):
        missing = sorted(set(group_cols) - set(tmp.columns))
        raise ValueError(f"Missing columns for interval width: {missing}")

    # Compute group means and SEM
    stat = (
        tmp.groupby(group_cols)["width_mean"]
        .agg(mean="mean", sem=_agg_sem)
        .reset_index()
    )

    if log_df:
        stat["degrees_of_freedom"] = np.log10(stat["degrees_of_freedom"])
    if log_width:
        # delta method approximation for SEM on log scale: sem/mean
        stat["sem"] = stat["sem"] / stat["mean"].replace(0, np.nan)
        stat["mean"] = np.log10(stat["mean"].replace(0, np.nan))

    # Create FacetGrid for plotting
    g = sns.FacetGrid(
        stat,
        row=aggregate_y,
        col=aggregate_x,
        margin_titles=False,
        sharey=False,
        sharex=True,
        height=height,
        aspect=aspect,
    )

    def draw(data, **kw):
        ax = plt.gca()
        for m in data["name"].unique():
            d = data[data["name"] == m].sort_values("degrees_of_freedom")
            if d.empty:
                continue
            # Plot groupwise mean interval width
            line = ax.plot(
                d["degrees_of_freedom"],
                d["mean"],
                marker="o",
                linestyle=linestyles.get(m, "-"),
                color=colors.get(m),
                label=m,
            )
            # Plot SE bands if requested
            if se_bands:
                ax.fill_between(
                    d["degrees_of_freedom"],
                    d["mean"] - d["sem"],
                    d["mean"] + d["sem"],
                    alpha=0.2,
                    color=line[0].get_color(),
                )

    g.map_dataframe(draw)

    # Remove default axes/text from subplots
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    # Set common axis labels once
    label_fs = 12
    g.axes[-1, g.axes.shape[1] // 2].set_xlabel(
        "Log Degrees of Freedom" if log_df else "Degrees of Freedom", fontsize=label_fs
    )
    g.axes[g.axes.shape[0] // 2, 0].set_ylabel(
        "Log Average CI Width" if log_width else "Average Interval Width",
        fontsize=label_fs,
    )

    # Set top column titles
    for j in range(g.axes.shape[1]):
        title = f"{aggregate_x if aggregate_x == 'SNR' else aggregate_x.replace('_', ' ').title()}: {g.col_names[j]}"
        g.axes[0, j].set_title(title)

    # Set right margin facet row labels
    for i in range(g.axes.shape[0]):
        txt = f"{aggregate_y if aggregate_y == 'SNR' else aggregate_y.replace('_', ' ').title()}: {g.row_names[i]}"
        g.axes[i, -1].text(
            1.02,
            0.95,
            txt,
            transform=g.axes[i, -1].transAxes,
            ha="left",
            va="top",
            rotation=90,
        )

    # Set overall figure title
    if log_width:
        sup = "Log Average CI Width vs " + (
            "Log Degrees of Freedom" if log_df else "Degrees of Freedom"
        )
    else:
        sup = "Average CI Width vs " + (
            "Log Degrees of Freedom" if log_df else "Degrees of Freedom"
        )
    g.fig.suptitle(sup, y=1.02)

    # Leave space for legend on the right
    g.fig.subplots_adjust(right=0.95)

    # Add legend to right of grid
    g.add_legend(
        title="Method",
        bbox_to_anchor=(0.85, 0.5),
        loc="center left",
        frameon=False,
        ncol=1,
    )

    # Save or display the plot
    if save_path:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return g


def expected_value_beta_rmse(gamma, sigma2=1, r2=5):
    return np.where(
        gamma < 1,
        sigma2 * gamma / (1 - gamma),
        r2 * (1 - 1 / gamma) + sigma2 / (1 - gamma),
    )


# fig, ax = plt.subplots()
# g_vals_1 = np.linspace(0.01, 0.99, 100)
# g_vals_2 = np.linspace(1.01, 2, 100)

# yg1 = expected_value_beta_rmse(g_vals_1)
# yg2 = expected_value_beta_rmse(g_vals_2)

# ax.plot(g_vals_1, expected_value_beta_rmse(g_vals_1), color="black")
# ax.plot(g_vals_2, expected_value_beta_rmse(g_vals_2), color="black")
# ax.vlines(1, ymin=yg2.min(), ymax=yg1.max(), colors="red", linestyles="dashed")
# plt.show()