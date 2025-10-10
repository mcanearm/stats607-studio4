import textwrap
from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from sklearn.metrics import mean_squared_error, r2_score

from studio7.src.simulation import generate_data


def _get_estimator_name(estimator_class):
    estimator_name = {
            LinearRegression: "OLS",
            QuantileRegressor: "QR",
            HuberRegressor: "Huber",
        }
    try:
        return estimator_name[estimator_class]
    except KeyError:
        # handle case when we use a partial function to set parameters
        try:
            return estimator_name[estimator_class.func]
        except (KeyError):
            raise ValueError(
                "Estimator not supported: options are LinearRegression, QuantileRegressor, HuberRegressor"
            )

class SimulationResult(object):
    def __init__(self, estimator, result_set) -> None:
        self.name = _get_estimator_name(estimator)
        self._df = pd.DataFrame(result_set)

    def __getattr__(self, name):
        # delegate all other attributes/methods to the underlying DataFrame
        # Prevent recursion during unpickling or before _df exists
        if "_df" not in self.__dict__:
            raise AttributeError(f"{name} not found")
        return getattr(self._df, name)
    
    def __str__(self):
        start_text = f"{'-' * 40}\nSim Result: {self.name}\nN_sim: {len(self.r2)}"
        param_text = f"p: {self.p[0]}, aspect_ratio: {self.aspect_ratio[0]}, degrees_of_freedom: {self.degrees_of_freedom[0]}, SNR: {self.SNR[0]}, rho: {self.rho[0]:0.2f}"
        rmse_ci_str = f"RMSE: {self._ci_string(self.rmse)}"
        r2_ci_str = f"R2: {self._ci_string(self.r2)}"
        coverage = f"Coverage (95%): {self.calculate_coverage()}"
        str_components = [start_text, param_text, rmse_ci_str, r2_ci_str, coverage]
        text_out = "\n".join(str_components)
        text_out += f"\n{'-' * 40}"

        return textwrap.dedent(text_out)

    @staticmethod
    def _ci_string(v):
        return f"({np.percentile(v, 2.5):0.3f}, {np.percentile(v, 97.5):0.3f})"

    def calculate_coverage(self, alpha=0.05):
        """Calculate coverage of the true values according to alpha using quantiles"""
        beta_hat_estimates = np.stack(self.beta_hat)
        se_betas = np.stack(self.se_beta)
        true_beta = np.stack(self.true_beta)

        t_stat = stats.t.ppf(1 - alpha / 2, df=self.N - 1)
        lower = beta_hat_estimates - t_stat[:, None] * se_betas
        upper = beta_hat_estimates + t_stat[:, None] * se_betas

        coverage = np.mean((true_beta >= lower) & (true_beta <= upper), axis=0)
        return coverage

    def save(self, output_dir: Path):
        output_filename = output_dir / self.filename
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "wb") as f:
            pkl.dump(self, f)

    @property
    def filename(self):
        return Path(
            f"{self.name}/p={self.p[0]}_snr={self.SNR[0]}_df={self.degrees_of_freedom[0]}_ar={self.aspect_ratio[0]}.pkl"
        )

    @classmethod
    def load(cls, input_filepath: Path):
        with open(input_filepath, "rb") as f:
            return pkl.load(f)


def run_simulation(estimator_class, n_sim=1000, **data_params) -> SimulationResult | dict:
    """
    Run the simulations for a given estimator class. Notably, we're just using
    the default parameters on each estimator class.
    """
    if n_sim > 1:
        outputs = [
            run_simulation(estimator_class, n_sim=1, **data_params)
            for _ in range(n_sim)
        ]
        return SimulationResult(estimator_class, outputs)
    else:
        # hacky increase to max iterations after seeing some issues with convergence
        random_state = (np.random.get_state(),)
        p = data_params["p"]  # this should error if p not provided
        rho = float(data_params["covariance"][0, 1]) # get any off diagonal term, they should be the same in our setup
        X, y, beta = generate_data(**data_params)
        model = estimator_class()
        preds = model.fit(X, y).predict(X)
        beta_hat = model.coef_
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        name = _get_estimator_name(estimator_class)  # just to validate

        N = X.shape[0]
        sigma_hat = np.sum((y - preds) ** 2) / (N - p)

        xtx_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(sigma_hat * np.diagonal(xtx_inv))

        return {
            "name": name,
            "random_state": random_state,
            "predictions": preds,
            "beta_hat": beta_hat,
            "rmse": rmse,
            "r2": r2,
            "true_beta": beta,
            "se_beta": se_beta,
            "N": N,
            "rho": rho,
            **data_params,
        }


if __name__ == "__main__":
    # Simple test
    sim_result = run_simulation(
        LinearRegression,
        n_sim=500,
        p=5,
        aspect_ratio=0.2,
        covariance=np.identity(5),
        degrees_of_freedom=100,
        SNR=2.0,
    )
    print(sim_result)
