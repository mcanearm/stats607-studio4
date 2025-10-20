import pickle as pkl
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.metrics import mean_squared_error

from studio8.src.simulation import generate_data


def _get_estimator_name(estimator_class):
    estimator_name = {
        LinearRegression: "OLS",
        QuantileRegressor: "QR",
        HuberRegressor: "Huber",
        Ridge: "OLS",
    }
    try:
        return estimator_name[estimator_class]
    except KeyError:
        # handle case when we use a partial function to set parameters
        try:
            return estimator_name[estimator_class.func]
        except KeyError:
            raise ValueError(
                "Estimator not supported: options are LinearRegression, QuantileRegressor, HuberRegressor"
            )


def ridge_sigma2_and_se(model: Ridge, X, y, sample_weight=None):
    """
    Given a fitted sklearn Ridge model, return:
      sigma2_hat, df_eff, rss, se_beta  (SEs aligned to model.coef_)
    Works for p > N. No refit; uses SVD of centered design.
    """
    y = np.asarray(y).ravel()
    y_hat = model.predict(X)
    resid = y - y_hat

    # (1) RSS
    if sample_weight is None:
        rss = float(resid @ resid)
        Xc = X.copy()
        if getattr(model, "fit_intercept", True):
            X_offset = getattr(model, "_X_offset", np.mean(X, axis=0))
            Xc = Xc - X_offset
    else:
        w = np.asarray(sample_weight).ravel()
        sw = np.sqrt(w)
        rss = float((resid * sw) @ (resid * sw))
        Xc = X.copy()
        if getattr(model, "fit_intercept", True):
            # weighted centering: subtract weighted mean
            X_offset = getattr(
                model, "_X_offset", (sw[:, None] * X).sum(axis=0) / sw.sum()
            )
            Xc = Xc - X_offset
        Xc = sw[:, None] * Xc  # apply weights to design for SVD/df

    # (2) SVD of centered (and weighted) X
    # Xc = U S V^T; shapes: U(N×r), S(r,), V(p×r), r = min(N, p)
    # We only need S and V (no need to compute U)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    # To get V we need the full SVD once; compute with UV but it's cheap relative to p,N
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # p×r

    alpha = float(model.intercept_)

    # (3) df_eff = sum s^2/(s^2+alpha)
    s2 = S**2
    df_eff = float(np.sum(s2 / (s2 + alpha)))

    # (4) sigma2_hat = RSS / (N - df_eff)
    N = X.shape[0]
    denom = N - df_eff
    # guard very small/negative denominators (return inf rather than crash)
    if denom <= 1e-12:
        sigma2_hat = np.inf
    else:
        sigma2_hat = rss / denom

    # (5) Var(beta_hat) diag via V and s:
    # Cov = sigma^2 * V diag(s^2/(s^2+alpha)^2) V^T
    w = s2 / (s2 + alpha) ** 2  # length r
    # diag(Cov) = sigma^2 * sum_k w_k * V_{jk}^2
    diag_cov = sigma2_hat * (V**2 @ w)  # shape (p,)
    se_beta = np.sqrt(diag_cov)

    return sigma2_hat, df_eff, rss, se_beta


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

    def __getitem__(self, key):
        """
        Allow use of normal pandas Dataframe indexing
        """
        out = self._df[key]
        if len(out[0]) > 1:
            return np.stack(out.values)
        else:
            return out

    def __str__(self):
        start_text = f"{'-' * 40}\nSim Result: {self.name}\nN_sim: {len(self.rmse)}"
        param_text = f"p: {self.p[0]}, aspect_ratio: {self.aspect_ratio[0]}, degrees_of_freedom: {self.degrees_of_freedom[0]}, SNR: {self.SNR[0]}, rho: {self.rho[0]}"
        rmse_ci_str = f"RMSE: {self._ci_string(self.rmse)}"
        coverage = f"Coverage (95%): {self.calculate_coverage()}"
        str_components = [start_text, param_text, rmse_ci_str, coverage]
        text_out = "\n".join(str_components)
        text_out += f"\n{'-' * 40}"

        return textwrap.dedent(text_out)

    @staticmethod
    def _ci_string(v):
        return f"({np.percentile(v, 2.5):0.3f}, {np.percentile(v, 97.5):0.3f})"

    def calculate_coverage(self, alpha=0.05):
        """Calculate coverage of the true values according to alpha using quantiles"""
        beta_hat_estimates = self["beta_hat"]
        se_betas = self["se_beta"]
        true_beta = self["true_beta"]

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
        return self._construct_filepath(
            self.name,
            p=self.p[0],
            # df=self.degrees_of_freedom[0],
            ar=self.aspect_ratio[0],
            # rho=self.rho[0],
            # sigma2=self.sigma2[0]
        )

    @classmethod
    def _construct_filepath(cls, name, **name_params):
        file_components = [f"{k}={v}" for k, v in name_params.items()]
        return Path(
            f"{name}/{'_'.join(file_components)}.pkl"
        )

    @classmethod
    def load(cls, input_filepath: Path):
        with open(input_filepath, "rb") as f:
            return pkl.load(f)


def run_simulation(
    estimator_class, n_sim=1000, rng=None, **data_params
) -> SimulationResult:
    """
    Run the simulations for a given estimator class. Notably, we're just using
    the default parameters on each estimator class.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _run_sim():
        p = data_params["p"]  # this should error if p not provided
        try:
            # get any off diagonal term, they should be the same in our setup
            rho = float(data_params["covariance"][0, 1])
        except IndexError:
            rho = np.nan
        X, y, beta = generate_data(rng=rng, **data_params)
        model = estimator_class()
        model.fit(X, y)
        preds = model.predict(X)
        beta_hat = model.coef_
        mse = mean_squared_error(beta, beta_hat)
        rmse = np.sqrt(mse)
        name = _get_estimator_name(estimator_class)
        N, p = X.shape

        try:
            resid = y - preds
            rss = float(resid @ resid)
            denom = max(1, N - p)  # guard to avoid division by zero/neg
            sigma2_hat = rss / denom
            xtx_inv = np.linalg.pinv(X.T @ X)
            se_beta = np.sqrt(np.clip(np.diag(xtx_inv), 0, np.inf) * sigma2_hat)
        except np.linalg.LinAlgError:
            sigma2_hat, df_eff, rss, se_beta = ridge_sigma2_and_se(model, X, y)

        alpha = 0.05
        tcrit = stats.t.ppf(1 - alpha / 2, df=N - 1)
        ci_lower = beta_hat - tcrit * se_beta
        ci_upper = beta_hat + tcrit * se_beta
        ci_width = ci_upper - ci_lower

        out = {
            "name": name,
            "random_state": rng,
            "predictions": preds,
            "beta_hat": beta_hat,
            "rmse": rmse,
            "mse": mse,
            "true_beta": beta,
            "se_beta": se_beta,
            "N": N,
            "rho": rho,
            "rng": rng,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width,
            **data_params,
        }
        return out

    outputs = SimulationResult(estimator_class, [_run_sim() for _ in range(n_sim)])
    return outputs


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
