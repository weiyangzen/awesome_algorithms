"""Minimal runnable MVP for tensor perturbations in cosmology.

We evolve each Fourier mode h_k(eta) with
    h'' + 2 (a'/a) h' + k^2 h = 0
on a radiation-to-matter transition background
    a(eta) = a_eq * [(eta/eta_eq)^2 + 2 (eta/eta_eq)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class BackgroundParams:
    """Background parameters in conformal time units."""

    eta_eq: float = 1.0
    a_eq: float = 1.0


def scale_factor_and_derivative(
    eta: np.ndarray | float,
    params: BackgroundParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, a') for the analytic radiation+matter interpolation."""
    eta_arr = np.asarray(eta, dtype=float)
    x = eta_arr / params.eta_eq
    a = params.a_eq * (x * x + 2.0 * x)
    a_prime = params.a_eq * (2.0 * x + 2.0) / params.eta_eq
    return a, a_prime


def conformal_hubble(eta: np.ndarray, params: BackgroundParams) -> np.ndarray:
    """Conformal Hubble scale a'/a sampled on eta."""
    a, a_prime = scale_factor_and_derivative(eta, params)
    return a_prime / a


def tensor_mode_rhs(
    eta: float,
    y: np.ndarray,
    k: float,
    params: BackgroundParams,
) -> np.ndarray:
    """RHS for y=[h, h'] with equation h'' + 2(a'/a)h' + k^2 h = 0."""
    h, h_prime = y
    a, a_prime = scale_factor_and_derivative(np.array([eta]), params)
    friction = 2.0 * (a_prime[0] / a[0])
    h_second = -friction * h_prime - (k * k) * h
    return np.array([h_prime, h_second], dtype=float)


def evolve_tensor_mode(
    k: float,
    eta_grid: np.ndarray,
    params: BackgroundParams,
    h_init: float = 1.0,
    h_prime_init: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically evolve one Fourier mode."""
    solution = solve_ivp(
        fun=lambda eta, y: tensor_mode_rhs(eta, y, k, params),
        t_span=(float(eta_grid[0]), float(eta_grid[-1])),
        y0=np.array([h_init, h_prime_init], dtype=float),
        t_eval=eta_grid,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed for k={k}: {solution.message}")

    h = solution.y[0]
    h_prime = solution.y[1]
    return h, h_prime


def estimate_horizon_entry_eta(
    k: float,
    eta_grid: np.ndarray,
    params: BackgroundParams,
) -> float:
    """Estimate eta_entry where k = a'/a by interpolation on sampled grid."""
    horizon = conformal_hubble(eta_grid, params)
    entered = k >= horizon

    if not np.any(entered):
        return float("nan")

    idx = int(np.argmax(entered))
    if idx == 0:
        return float(eta_grid[0])

    eta_l, eta_r = eta_grid[idx - 1], eta_grid[idx]
    h_l, h_r = horizon[idx - 1], horizon[idx]
    if np.isclose(h_l, h_r):
        return float(eta_r)

    w = (k - h_l) / (h_r - h_l)
    return float(eta_l + w * (eta_r - eta_l))


def adiabatic_invariant_proxy(
    h: np.ndarray,
    h_prime: np.ndarray,
    k: float,
    eta_grid: np.ndarray,
    params: BackgroundParams,
) -> np.ndarray:
    """WKB-like invariant proxy for subhorizon oscillatory modes."""
    a, _ = scale_factor_and_derivative(eta_grid, params)
    return (a * a * (h_prime * h_prime + (k * k) * (h * h))) / (2.0 * k)


def run_mvp() -> tuple[pd.DataFrame, float, float, float]:
    """Run all modes and return summary table + key diagnostics."""
    params = BackgroundParams(eta_eq=1.0, a_eq=1.0)
    eta_grid = np.linspace(0.05, 50.0, 4000)
    k_values = np.array([0.02, 0.05, 0.10, 0.30, 1.0, 3.0, 10.0], dtype=float)

    rows: list[dict[str, float]] = []
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    horizon_end = float(conformal_hubble(np.array([eta_grid[-1]]), params)[0])

    for k in k_values:
        h, h_prime = evolve_tensor_mode(k=k, eta_grid=eta_grid, params=params)
        cache[float(k)] = (h, h_prime)

        eta_entry = estimate_horizon_entry_eta(k=k, eta_grid=eta_grid, params=params)
        transfer_abs = float(abs(h[-1] / h[0]))
        omega_gw_proxy = float((k * k) * (h[-1] * h[-1]) / (12.0 * horizon_end * horizon_end))

        rows.append(
            {
                "k": float(k),
                "eta_entry": eta_entry,
                "h_initial": float(h[0]),
                "h_final": float(h[-1]),
                "transfer_abs": transfer_abs,
                "omega_gw_proxy": omega_gw_proxy,
            }
        )

    summary = pd.DataFrame(rows).sort_values(by="k").reset_index(drop=True)

    low_k_mean = float(summary.loc[summary["k"] <= 0.10, "transfer_abs"].mean())
    high_k_mean = float(summary.loc[summary["k"] >= 1.0, "transfer_abs"].mean())

    h_hi, h_prime_hi = cache[10.0]
    invariant = adiabatic_invariant_proxy(
        h=h_hi,
        h_prime=h_prime_hi,
        k=10.0,
        eta_grid=eta_grid,
        params=params,
    )
    late_mask = eta_grid >= 10.0
    invariant_cv = float(np.std(invariant[late_mask]) / np.mean(invariant[late_mask]))

    entered = summary.dropna(subset=["eta_entry"]).copy()
    eta_entries = entered["eta_entry"].to_numpy(dtype=float)

    # Assertion 1: the smallest-k mode remains superhorizon in this finite time window.
    k002_row = summary.loc[np.isclose(summary["k"], 0.02)].iloc[0]
    assert np.isnan(k002_row["eta_entry"]), "k=0.02 should not enter the horizon by eta_max."
    assert float(k002_row["transfer_abs"]) > 0.85, "Superhorizon mode should remain nearly frozen."

    # Assertion 2: high-k modes are much more damped than low-k modes.
    assert high_k_mean < 0.05 * low_k_mean, "Expected stronger damping for high-k modes."

    # Assertion 3: larger k should enter the horizon earlier.
    assert np.all(np.diff(eta_entries) < 0.0), "Horizon-entry ordering by k is incorrect."

    # Assertion 4: subhorizon adiabatic invariant is approximately conserved for k=10.
    assert invariant_cv < 0.02, "Adiabatic invariant fluctuates too much for k=10."

    return summary, low_k_mean, high_k_mean, invariant_cv


def main() -> None:
    summary, low_k_mean, high_k_mean, invariant_cv = run_mvp()

    display = summary.copy()
    display["eta_entry"] = display["eta_entry"].map(
        lambda x: "not-entered" if np.isnan(x) else f"{x:.3f}"
    )

    print("Tensor perturbation MVP: h'' + 2(a'/a)h' + k^2 h = 0")
    print(display.to_string(index=False))
    print()
    print(f"mean transfer (k <= 0.1): {low_k_mean:.6e}")
    print(f"mean transfer (k >= 1.0): {high_k_mean:.6e}")
    print(f"k=10 adiabatic-invariant CV (eta>=10): {invariant_cv:.6e}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
