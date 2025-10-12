"""Generate plots"""

from pathlib import Path
from studio7.src.plot_function import plot_grid, load_data, plot_boxplot, plot_individual


if __name__ == "__main__":
    figure_dir = Path("studio7/figures/")
    output_dir = Path("studio7/sim_outputs/")
    
    print("Loading data...")
    results_combined = load_data(base_path=output_dir)
    
    # line plots
    print("Generating line plots...")
    col_list = ["aspect_ratio", "SNR", "rho"]
    for i in range(len(col_list)):
        for j in range(i + 1, len(col_list)):
            g = plot_grid(results_combined, height=1.3, aspect=1.3, log_df=True, 
                        aggregate_x=col_list[i], aggregate_y=col_list[j],
                        save_path=f"studio7/figures/grids/grid_{col_list[i]}_{col_list[j]}")
            plot_individual(results_combined, height=3.5, aspect=1.3, log_df=True, 
                        aggregate_x=col_list[i], aggregate_y=col_list[j],
                        save_path=f"studio7/figures/individuals/individual")
    
    col_list = ["degrees_of_freedom", "aspect_ratio", "SNR", "rho"]
    for i in range(len(col_list)):
        plot_boxplot(results_combined, height=4, aspect=1.3, log_rmse=True, log_x=False,
                    x=col_list[i], n_boxplots=5,
                    save_path=f"studio7/figures/boxplots/boxplot_{col_list[i]}")
    

