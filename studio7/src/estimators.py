import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from sklearn.metrics import mean_squared_error, r2_score

from studio7.src.simulation import generate_data, make_positive_definite


def _get_estimator_name(estimator_class):
    try:
        estimator_name = {
            LinearRegression: "ols",
            QuantileRegressor: "quantile",
            HuberRegressor: "huber",
        }[estimator_class]
        return estimator_name
    except KeyError:
        raise ValueError(
            "Estimator not supported: options are LinearRegression, QuantileRegressor, HuberRegressor"
        )


class SimulationResult(object):
    def __init__(self, estimator, result_set) -> None:
        self.name = _get_estimator_name(estimator)
        self._df = pd.DataFrame(result_set)

    def __getattr__(self, name):
        # delegate all other attributes/methods to the underlying DataFrame
        return getattr(self._df, name)

    def __str__(self):
        rmse_ci_str = f"RMSE: {self._ci_string(self.rmse)}"
        r2_ci_str = f"R2: {self._ci_string(self.r2)}"
        start_text = f"{'-' * 40}\nSim Result: {self.name}\nN_sim: {len(self.r2)}"
        coverage = f"Coverage (95%): {self.calculate_coverage()}"
        str_components = [start_text, rmse_ci_str, r2_ci_str, coverage]
        text_out = "\n".join(str_components)
        text_out += f"\n{'-' * 40}"

        return textwrap.dedent(text_out)

    @staticmethod
    def _ci_string(v):
        return f"({np.percentile(v, 2.5):0.3f}, {np.percentile(v, 97.5)})"

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

    def save(self, outpur_dir: Path):
        output_filename = (
            outpur_dir
            / f"{self.name}/p={self.p[0]}_snr={self.SNR[0]}_df={self.degrees_of_freedom[0]}_ar={self.aspect_ratio[0]}.csv"
        )
        self._df.to_csv(output_filename, index=False)

    @classmethod
    def load(cls, input_filepath: Path):
        df = pd.read_csv(input_filepath)
        estimator_name = input_filepath.stem.split("_")[0]
        estimator_class = {
            "ols": LinearRegression,
            "quantile": QuantileRegressor,
            "huber": HuberRegressor,
        }[estimator_name]
        return cls(estimator_class, df.to_dict(orient="records"))


def run_simulation(estimator_class, n_sim=1000, **data_params):
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
        random_state = (np.random.get_state(),)
        p = data_params["p"]  # this should error if p not provided
        X, y, beta = generate_data(**data_params)
        model = estimator_class()
        preds = model.fit(X, y).predict(X)
        beta_hat = model.coef_
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        name = _get_estimator_name(estimator_class)  # just to validate

        N = X.shape[0]
        sigma_hat = np.sum((y - preds) ** 2) / (N - p)

        xtx_inv = make_positive_definite(np.linalg.inv(X.T @ X))
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
            **data_params,
        }
