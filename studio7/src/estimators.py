import numpy as np
import textwrap
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from studio7.src.simulation import generate_data


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

    def __str__(self):
        rmse_ci_str = f"RMSE: {self._ci_string(self.rmse)}"
        r2_ci_str = f"R2: {self._ci_string(self.r2)}"
        start_text = f"{'-' * 40}\nSim Result: {self.name}\nN_sim: {len(self.r2)}"
        coverage = f"Coverage (95%): {np.mean(self.calculate_coverage()):0.3f}"
        str_components = [start_text, rmse_ci_str, r2_ci_str, coverage]
        text_out = "\n".join(str_components)
        text_out += f"\n{'-' * 40}"

        return textwrap.dedent(text_out)

    @staticmethod
    def _ci_string(v):
        return f"({np.percentile(v, 2.5):0.3f}, {np.percentile(v, 97.5):0.3f})"

    def calculate_coverage(self, alpha=0.05):
        """Calculate coverage of the true values according to alpha using quantiles"""

        self

        coverage = np.mean((self.true_betas >= lower) & (self.true_betas <= upper), axis=0)
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
        X_with_intercept = np.empty(shape=(N, p), dtype=np.dtype("float"))
        X_with_intercept[:, 0] = 1
        X_with_intercept[:, 1:p] = X[:, 1:]

        sigma_hat = np.sum((y - preds) ** 2)/(N-p)
        se_beta = np.sqrt(sigma_hat) * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_beta[np.diag_indices_from(se_beta)]
        

        return {
            "predictions": preds,
            "beta_hat": beta_hat,
            "rmse": rmse,
            "r2": r2,
            "true_beta": beta,
            **data_params,
        }


p = 30
print(
    run_simulation(
        LinearRegression,
        n_sim=200,
        p=p,
        aspect_ratio=0.5,
        covariance=np.identity(p),
        degrees_of_freedom=5,
        SNR=5,
    )
)
