"""Generate plots"""
import argparse

from studio7.src.plot_function import (
    plot_grid,
    load_data,
    plot_boxplot,
    plot_individual,
    plot_coverage,
    plot_interval_width,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from simulation results")
    parser.add_argument(
        "--figures_dir", type=str, default="studio7/figures/", help="Directory to save figures"
    )
    parser.add_argument(
        "--output_dir", type=str, default="studio7/sim_outputs/", help="Directory of simulation outputs"
    )
    parser.add_argument(
        "--log-rmse", action="store_true", help="Whether to log-transform RMSE in boxplots", default=False
    )

    # this is always on, but... maybe we change that later.
    parser.add_argument(
        "--log-x", action="store_true", help="Whether to log-transform degrees_of_freedom in plots", default=False
    )

    parser.add_argument("--ci-level", type=float, default=0.95,
                    help="Target CI level for coverage reference line")
    parser.add_argument("--log-width", action="store_true", default=False,
                        help="Plot log of average interval width")
    parser.add_argument("--no-bands", action="store_true", default=False,
                        help="Disable SEM shading bands in line plots")


    args = parser.parse_args()
    figure_dir = args.figures_dir
    output_dir = args.output_dir

    print("Loading data...")
    results_combined = load_data(base_path=output_dir)

    # line plots
    print("Generating line plots...")
    col_list = ["aspect_ratio", "SNR", "rho"]
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            g = plot_grid(
                results_combined,
                height=1.3,
                aspect=1.3,
                log_df=True,
                aggregate_x=col_list[i],
                aggregate_y=col_list[j],
                save_path=f"studio7/figures/grids/grid_{col_list[i]}_{col_list[j]}",
            )
            plot_individual(
                results_combined,
                height=3.5,
                aspect=1.3,
                log_df=True,
                aggregate_x=col_list[i],
                aggregate_y=col_list[j],
                save_path="studio7/figures/individuals/individual",
            )

    col_list = ["degrees_of_freedom", "aspect_ratio", "SNR", "rho"]
    for i in range(len(col_list)):
        plot_boxplot(
            results_combined,
            height=4,
            aspect=1.3,
            log_rmse=args.log_rmse,
            log_x=args.log_x,
            x=col_list[i],
            n_boxplots=5,
            save_path=f"studio7/figures/boxplots/boxplot_{col_list[i]}",
        )
    
     # ========= coverage grids =========
    print("Generating coverage grids...")
    col_list = ["aspect_ratio", "SNR", "rho"]
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            plot_coverage(
                results_combined,
                aggregate_x=col_list[i],
                aggregate_y=col_list[j],
                log_df=True,                         
                ci_level=args.ci_level,
                se_bands=not args.no_bands,
                height=1.3,
                aspect=1.3,
                save_path=str(f"studio7/figures/coverage_grids/coverage_{col_list[i]}_{col_list[j]}"),
            )

    # ========= interval width grids =========
    print("Generating interval-width grids...")
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            plot_interval_width(
                results_combined,
                aggregate_x=col_list[i],
                aggregate_y=col_list[j],
                log_df=True,                         # 与上面一致
                log_width=args.log_width,
                se_bands=not args.no_bands,
                height=1.3,
                aspect=1.3,
                save_path=str(f"studio7/figures/interval_width_grids/interval_width_{col_list[i]}_{col_list[j]}"),
            )

    print(f"Done. Figures saved under: {figure_dir}")