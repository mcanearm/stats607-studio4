import numpy as np
import textwrap
from sklearn.linear_model import LinearRegression, QuantileRegressor, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score


def generate_data(*args, **kwargs):
    X = np.random.normal(size=(100, 3))
    y = np.random.normal(size=100)
    return X, y


class SimulationResult:
    def __init__(self, estimator, result_set) -> None:
        self.name = {
            LinearRegression: "OLS",
            QuantileRegressor: "QR",
            HuberRegressor: "Huber",
        }[estimator]
        self.betas = np.array([res["beta_hat"] for res in result_set])
        self.rmse = np.array([res["rmse"] for res in result_set])
        self.r2 = np.array([res["r2"] for res in result_set])
        self.predictions = np.array([res["predictions"] for res in result_set])

    def __str__(self):
        rmse_ci_str = f"RMSE: {self._ci_string(self.rmse)}"
        r2_ci_str = f"R2: {self._ci_string(self.r2)}"
        betas_ci_str = [
            f"\tBeta {i+1}: {self._ci_string(self.betas[:, i])}"
            for i in range(self.betas.shape[1])
        ]
        start_text = (
            f"{'-'*40}\nSim Result: {self.name}\nN_sim: {len(self.r2)}"
        )
        str_components = [start_text, "Betas:", *betas_ci_str, rmse_ci_str, r2_ci_str] 
        text_out = "\n".join(str_components)
        text_out += f"\n{'-'*40}"
            
        # RMSE_CI: {np.percentile(self.rmse, 2.5):0.3f} - {np.percentile(self.rmse, 97.5):0.3f}
        # R2_CI: {np.percentile(self.r2, 2.5):0.3f} - {np.percentile(self.r2, 97.5):0.3f}
        return textwrap.dedent(text_out)

    @staticmethod
    def _ci_string(v):
        return f"({np.percentile(v, 2.5):0.3f}, {np.percentile(v, 97.5):0.3f})"


def run_simulation(estimator_class, n_sim=1000, **data_params):
    if n_sim > 1:
        outputs = [
            run_simulation(estimator_class, n_sim=1, **data_params)
            for _ in range(n_sim)
        ]
        return SimulationResult(estimator_class, outputs)
    else:
        X, y = generate_data(**data_params)
        model = estimator_class()
        preds = model.fit(X, y).predict(X)
        beta_hat = model.coef_
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        return {"predictions": preds, "beta_hat": beta_hat, "rmse": rmse, "r2": r2}

print(run_simulation(LinearRegression, n_sim=100))
