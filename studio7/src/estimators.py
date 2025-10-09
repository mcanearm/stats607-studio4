import numpy as np
import textwrap
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from studio7.src.simulation import generate_data
from scipy import stats
from studio7.src.simulation import make_positive_definite


class SimulationResult:
    def __init__(self, estimator, result_set) -> None:
        self.name = {
            LinearRegression: "OLS",
            QuantileRegressor: "QR",
            HuberRegressor: "Huber",
        }[estimator]
        self.beta_hat = np.array([res["beta_hat"] for res in result_set])
        self.rmse = np.array([res["rmse"] for res in result_set])
        self.r2 = np.array([res["r2"] for res in result_set])
        self.predictions = np.array([res["predictions"] for res in result_set])
        self.true_betas = np.array([res["true_beta"] for res in result_set])
        self.se_beta = np.array([res["se_beta"] for res in result_set])
        self.N = np.array([res["N"] for res in result_set])
        self.p = len(self.true_betas[0])

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
        t_stat = stats.t.ppf(1 - alpha / 2, df=self.N - 1)
        lower = self.beta_hat - t_stat[:, None] * self.se_beta
        upper = self.beta_hat + t_stat[:, None] * self.se_beta

        coverage = np.mean(
            (self.true_betas >= lower) & (self.true_betas <= upper), axis=0
        )
        return coverage


def run_simulation(estimator_class, n_sim=1000, **data_params):
    if n_sim > 1:
        outputs = [
            run_simulation(estimator_class, n_sim=1, **data_params)
            for _ in range(n_sim)
        ]
        return SimulationResult(estimator_class, outputs)
    else:
        X, y, beta = generate_data(**data_params)
        model = estimator_class()
        preds = model.fit(X, y).predict(X)
        beta_hat = model.coef_
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        N = X.shape[0]
        sigma_hat = np.sum((y - preds) ** 2) / (N - p)

        xtx_inv = make_positive_definite(np.linalg.inv(X.T @ X))
        se_beta = np.diagonal(np.sqrt(sigma_hat * xtx_inv))

        return {
            "predictions": preds,
            "beta_hat": beta_hat,
            "rmse": rmse,
            "r2": r2,
            "true_beta": beta,
            "se_beta": se_beta,
            "N": N,
            **data_params,
        }


if __name__ == "__main__":
    p = 10
    for regressor in [LinearRegression, QuantileRegressor, HuberRegressor]:
        sim_result = run_simulation(
            regressor,
            n_sim=200,
            p=p,
            aspect_ratio=0.2,
            covariance=np.identity(p),
            degrees_of_freedom=100,
            SNR=2.0,
        )
        print(sim_result)
