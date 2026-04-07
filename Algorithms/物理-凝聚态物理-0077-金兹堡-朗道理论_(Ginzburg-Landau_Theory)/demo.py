"""Minimal runnable MVP for Ginzburg-Landau theory (PHYS-0077).

This script solves a 1D time-dependent Ginzburg-Landau relaxation model
(without magnetic vector potential) on a finite-difference grid:

    d psi / dt = eps * psi - beta * psi^3 + xi^2 * d^2 psi / dx^2

where eps = 1 - T/Tc.

For uniform equilibrium and eps > 0, theory predicts:

    |psi|_eq = sqrt(eps / beta)

The MVP runs a temperature sweep across Tc, compares numerical and analytic
order parameter amplitudes, performs a near-Tc linear scaling fit with
scikit-learn, and uses PyTorch to invert beta from simulated data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson
from scipy.sparse import csr_matrix, diags
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class GLConfig:
    """Configuration for the 1D dimensionless GL relaxation simulation."""

    tc: float = 1.0
    beta: float = 1.0
    xi: float = 0.2
    length: float = 20.0
    n_grid: int = 256
    dt: float = 0.04
    max_steps: int = 6000
    tol: float = 2.0e-6
    seed: int = 20260407
    temperatures: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05)

    def validate(self) -> None:
        if self.tc <= 0 or self.beta <= 0 or self.xi <= 0:
            raise ValueError("tc, beta, xi must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.n_grid < 32:
            raise ValueError("n_grid must be >= 32")
        if self.max_steps < 200:
            raise ValueError("max_steps should be >= 200")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if len(self.temperatures) < 4:
            raise ValueError("need at least 4 temperature samples")


@dataclass(frozen=True)
class SimulationResult:
    """Numerical result for one temperature."""

    temperature: float
    epsilon: float
    order_param_abs_mean: float
    order_param_sq_mean: float
    analytic_order_abs: float
    free_energy_density: float
    converged_steps: int


def build_neumann_laplacian(n: int, dx: float) -> csr_matrix:
    """Build 1D second-derivative matrix with Neumann boundary condition."""
    main = -2.0 * np.ones(n, dtype=np.float64)
    upper = np.ones(n - 1, dtype=np.float64)
    lower = np.ones(n - 1, dtype=np.float64)

    # Ghost-point elimination for Neumann BC: psi_{-1}=psi_1, psi_{N}=psi_{N-2}.
    upper[0] = 2.0
    lower[-1] = 2.0

    lap = diags((lower, main, upper), offsets=(-1, 0, 1), shape=(n, n), dtype=np.float64)
    return (lap / (dx * dx)).tocsr()


def analytic_uniform_order_parameter(epsilon: float, beta: float) -> float:
    """Return the uniform GL equilibrium amplitude for a given epsilon."""
    if epsilon <= 0.0:
        return 0.0
    return float(np.sqrt(epsilon / beta))


def compute_free_energy_density(psi: np.ndarray, x: np.ndarray, epsilon: float, cfg: GLConfig) -> float:
    """Compute average free-energy density for current order-parameter field."""
    grad = np.gradient(psi, x, edge_order=2)
    density = -epsilon * psi * psi + 0.5 * cfg.beta * psi**4 + (cfg.xi * cfg.xi) * grad * grad
    total = float(simpson(density, x=x))
    return total / cfg.length


def relax_one_temperature(
    cfg: GLConfig,
    x: np.ndarray,
    laplacian: csr_matrix,
    temperature: float,
    rng: np.random.Generator,
) -> SimulationResult:
    """Integrate TDGL relaxation for one temperature and return summary metrics."""
    epsilon = 1.0 - temperature / cfg.tc

    # Positive bias avoids artificial domain walls in this teaching-level MVP.
    psi = 0.05 + 0.01 * rng.standard_normal(cfg.n_grid)

    converged_steps = cfg.max_steps
    for step in range(cfg.max_steps):
        lap_term = laplacian @ psi
        dpsi = cfg.dt * (epsilon * psi - cfg.beta * psi**3 + (cfg.xi * cfg.xi) * lap_term)
        psi_next = psi + dpsi

        if not np.isfinite(psi_next).all():
            raise FloatingPointError("non-finite values encountered during TDGL update")

        max_delta = float(np.max(np.abs(psi_next - psi)))
        psi = psi_next
        if step > 200 and max_delta < cfg.tol:
            converged_steps = step + 1
            break

    order_abs = float(np.mean(np.abs(psi)))
    order_sq = float(np.mean(psi * psi))
    analytic_abs = analytic_uniform_order_parameter(epsilon, cfg.beta)
    free_density = compute_free_energy_density(psi, x, epsilon, cfg)

    return SimulationResult(
        temperature=float(temperature),
        epsilon=float(epsilon),
        order_param_abs_mean=order_abs,
        order_param_sq_mean=order_sq,
        analytic_order_abs=analytic_abs,
        free_energy_density=free_density,
        converged_steps=int(converged_steps),
    )


def fit_near_tc_scaling(results_df: pd.DataFrame) -> tuple[float, float, float]:
    """Fit <|psi|^2> = slope * epsilon + intercept near Tc (epsilon > 0 small)."""
    mask = (results_df["epsilon"] > 0.0) & (results_df["epsilon"] <= 0.15)
    subset = results_df.loc[mask]
    if len(subset) < 3:
        raise ValueError("not enough points below Tc for linear scaling fit")

    x = subset[["epsilon"]].to_numpy(dtype=np.float64)
    y = subset["order_param_sq_mean"].to_numpy(dtype=np.float64)

    model = LinearRegression()
    model.fit(x, y)
    r2 = float(model.score(x, y))

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    return slope, intercept, r2


def torch_fit_beta(results_df: pd.DataFrame, epochs: int = 600, lr: float = 0.05) -> tuple[float, float]:
    """Use PyTorch to infer beta from simulated epsilon vs <|psi|^2> data."""
    mask = results_df["epsilon"] > 0.0
    subset = results_df.loc[mask]
    if len(subset) < 3:
        raise ValueError("not enough superconducting points for beta inversion")

    eps = torch.tensor(subset["epsilon"].to_numpy(dtype=np.float32))
    obs = torch.tensor(subset["order_param_sq_mean"].to_numpy(dtype=np.float32))

    raw = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        beta_hat = torch.nn.functional.softplus(raw) + 1.0e-6
        pred = eps / beta_hat
        loss = torch.mean((pred - obs) ** 2)
        loss.backward()
        optimizer.step()

    beta_est = float((torch.nn.functional.softplus(raw) + 1.0e-6).detach().cpu().item())
    final_loss = float(loss.detach().cpu().item())
    return beta_est, final_loss


def main() -> None:
    cfg = GLConfig()
    cfg.validate()

    x = np.linspace(-0.5 * cfg.length, 0.5 * cfg.length, cfg.n_grid, dtype=np.float64)
    dx = float(x[1] - x[0])
    laplacian = build_neumann_laplacian(cfg.n_grid, dx)

    rng = np.random.default_rng(cfg.seed)

    results = []
    for temp in cfg.temperatures:
        results.append(relax_one_temperature(cfg, x, laplacian, temp, rng))

    df = pd.DataFrame([r.__dict__ for r in results])
    df["abs_error_vs_analytic"] = np.abs(df["order_param_abs_mean"] - df["analytic_order_abs"])
    analytic_abs = df["analytic_order_abs"].to_numpy(dtype=np.float64)
    rel_error = np.full(len(df), np.nan, dtype=np.float64)
    mask = analytic_abs > 1.0e-10
    rel_error[mask] = df.loc[mask, "abs_error_vs_analytic"].to_numpy(dtype=np.float64) / analytic_abs[mask]
    df["rel_error_vs_analytic"] = rel_error

    slope, intercept, r2 = fit_near_tc_scaling(df)
    beta_est, torch_mse = torch_fit_beta(df)

    pd.set_option("display.width", 140)
    pd.set_option("display.precision", 8)

    print("=== Ginzburg-Landau Theory MVP (1D TDGL relaxation) ===")
    print(
        f"grid={cfg.n_grid}, length={cfg.length}, dt={cfg.dt}, "
        f"temperatures={len(cfg.temperatures)}, beta_true={cfg.beta:.4f}"
    )
    print(df.to_string(index=False))

    print("\n--- Near-Tc scaling fit: <|psi|^2> = slope * epsilon + intercept ---")
    print(f"slope={slope:.6f}, intercept={intercept:.6f}, r2={r2:.6f}")

    print("\n--- PyTorch inversion of beta from <|psi|^2> data ---")
    print(f"beta_est={beta_est:.6f}, torch_mse={torch_mse:.6e}")

    # Acceptance checks for this MVP.
    normal_mask = df["epsilon"] <= 0.0
    sc_mask = df["epsilon"] > 0.0

    normal_max = float(df.loc[normal_mask, "order_param_abs_mean"].max()) if normal_mask.any() else 0.0
    sc_rel_err_mean = float(df.loc[sc_mask, "rel_error_vs_analytic"].mean()) if sc_mask.any() else 0.0

    finite_cols = [
        "temperature",
        "epsilon",
        "order_param_abs_mean",
        "order_param_sq_mean",
        "analytic_order_abs",
        "free_energy_density",
        "converged_steps",
        "abs_error_vs_analytic",
    ]
    assert np.isfinite(df[finite_cols].to_numpy(dtype=np.float64)).all(), "non-finite values in summary table"
    assert normal_max < 0.06, "normal-state order parameter should be near zero"
    assert sc_rel_err_mean < 0.08, "superconducting-state amplitude deviates too much from analytic GL value"
    assert r2 > 0.95, "near-Tc scaling fit quality is too low"
    assert abs(beta_est - cfg.beta) < 0.15, "torch-inverted beta deviates too much from ground truth"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
