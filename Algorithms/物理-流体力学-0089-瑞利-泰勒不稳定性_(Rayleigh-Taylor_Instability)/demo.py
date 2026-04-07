"""Minimal runnable MVP for Rayleigh-Taylor instability (linear 2-layer interface model)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class RTIConfig:
    """Configuration for a 2-layer incompressible inviscid RTI linear model."""

    rho_top: float = 2.0
    rho_bottom: float = 1.0
    gravity: float = 9.81
    surface_tension: float = 0.06
    domain_length: float = 2.0 * np.pi
    max_mode: int = 16
    nx: int = 512
    nt: int = 260
    time_end: float = 0.35
    eta0_scale: float = 2.0e-7
    random_seed: int = 19

    @property
    def rho_sum(self) -> float:
        return self.rho_top + self.rho_bottom

    @property
    def atwood(self) -> float:
        return (self.rho_top - self.rho_bottom) / self.rho_sum


def dispersion_discriminant(k: np.ndarray, cfg: RTIConfig) -> np.ndarray:
    """Return the linear RTI growth discriminant D(k) = gamma^2 for unstable modes.

    For two inviscid incompressible layers with heavy fluid on top:
    gamma^2(k) = A*g*k - sigma*k^3/(rho_top + rho_bottom)

    where A is the Atwood number.
    - D(k) > 0: unstable, growth rate gamma = sqrt(D(k))
    - D(k) <= 0: stable (oscillatory under capillary-gravity balance)
    """

    buoyancy_drive = cfg.atwood * cfg.gravity * k
    capillary_stabilization = cfg.surface_tension * (k**3) / cfg.rho_sum
    return buoyancy_drive - capillary_stabilization


def mode_table(cfg: RTIConfig) -> pd.DataFrame:
    """Build modal properties for m = 1..max_mode."""

    modes = np.arange(1, cfg.max_mode + 1, dtype=int)
    k = 2.0 * np.pi * modes / cfg.domain_length
    disc = dispersion_discriminant(k, cfg)

    growth = np.sqrt(np.clip(disc, a_min=0.0, a_max=None))
    omega_stable = np.sqrt(np.clip(-disc, a_min=0.0, a_max=None))

    df = pd.DataFrame(
        {
            "mode": modes,
            "k": k,
            "discriminant": disc,
            "growth_rate": growth,
            "omega_stable": omega_stable,
        }
    )
    df["is_unstable"] = df["growth_rate"] > 0.0
    df["e_folding_time"] = np.where(df["growth_rate"] > 0.0, 1.0 / df["growth_rate"], np.inf)
    return df


def find_unstable_intervals(
    cfg: RTIConfig, k_min: float = 1.0e-4, k_max: float = 60.0, grid_size: int = 3200
) -> List[Tuple[float, float]]:
    """Locate unstable k-intervals where D(k) > 0 using sign bracketing + brentq."""

    grid = np.linspace(k_min, k_max, grid_size)
    values = dispersion_discriminant(grid, cfg)

    roots: List[float] = []
    for i in range(grid_size - 1):
        left_k, right_k = float(grid[i]), float(grid[i + 1])
        left_v, right_v = float(values[i]), float(values[i + 1])

        if left_v * right_v < 0.0:
            root = brentq(lambda z: float(dispersion_discriminant(np.array([z]), cfg)[0]), left_k, right_k)
            roots.append(float(root))

    roots = sorted({round(r, 10) for r in roots if k_min <= r <= k_max})

    boundaries: List[float] = [k_min] + roots + [k_max]
    intervals: List[Tuple[float, float]] = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        if b - a <= 1.0e-10:
            continue
        mid = 0.5 * (a + b)
        if float(dispersion_discriminant(np.array([mid]), cfg)[0]) > 0.0:
            intervals.append((a, b))

    return intervals


def make_initial_mode_coefficients(cfg: RTIConfig, mode_ids: Sequence[int]) -> np.ndarray:
    """Create deterministic small-amplitude complex modal coefficients."""

    rng = np.random.default_rng(cfg.random_seed)
    mode_ids = np.asarray(mode_ids, dtype=float)

    amplitudes = cfg.eta0_scale / np.power(mode_ids, 1.4)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=mode_ids.shape[0])
    coeffs = amplitudes * np.exp(1j * phases)
    return coeffs


def simulate_interface(df: pd.DataFrame, cfg: RTIConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate eta(x,t) by linear modal superposition."""

    modes = df["mode"].to_numpy(dtype=int)
    k = df["k"].to_numpy(dtype=float)
    gamma = df["growth_rate"].to_numpy(dtype=float)
    omega = df["omega_stable"].to_numpy(dtype=float)

    x = np.linspace(0.0, cfg.domain_length, cfg.nx, endpoint=False)
    t = np.linspace(0.0, cfg.time_end, cfg.nt)

    coeff0 = make_initial_mode_coefficients(cfg, modes)

    temporal = np.exp(np.outer(t, gamma + 1j * omega))
    spatial = np.exp(1j * np.outer(k, x))
    eta_complex = temporal[:, :, None] * coeff0[None, :, None] * spatial[None, :, :]
    eta = np.real(np.sum(eta_complex, axis=1))

    return x, t, eta, coeff0


def estimate_growth_from_fft(
    eta: np.ndarray, t: np.ndarray, dominant_mode: int
) -> Tuple[float, float, float, np.ndarray]:
    """Estimate dominant modal growth via FFT amplitude and linear regression."""

    fft_values = np.fft.rfft(eta, axis=1)
    amp = (2.0 / eta.shape[1]) * np.abs(fft_values[:, dominant_mode])

    fit_mask = (t >= 0.20 * t[-1]) & (amp > 1.0e-18)
    x_fit = t[fit_mask].reshape(-1, 1)
    y_fit = np.log(amp[fit_mask])

    if x_fit.shape[0] < 6:
        raise RuntimeError("Insufficient points for growth-rate regression fit.")

    reg = LinearRegression()
    reg.fit(x_fit, y_fit)
    y_pred = reg.predict(x_fit)

    slope = float(reg.coef_[0])
    fit_r2 = float(r2_score(y_fit, y_pred))
    fit_mae = float(mean_absolute_error(y_fit, y_pred))
    return slope, fit_r2, fit_mae, amp


def integrate_scalar_growth_ode(gamma: float, amplitude0: float, t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Integrate dA/dt = gamma*A and compare with analytic exponential growth."""

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        return np.array([gamma * y[0]], dtype=float)

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        y0=np.array([amplitude0], dtype=float),
        t_eval=t,
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    numeric = sol.y[0]
    analytic = amplitude0 * np.exp(gamma * (t - t[0]))
    rel_err = np.max(np.abs(numeric - analytic) / np.maximum(np.abs(analytic), 1.0e-18))
    return numeric, float(rel_err)


def torch_energy_growth_rate(eta: np.ndarray, t: np.ndarray) -> float:
    """Estimate growth from interface energy E ~ <eta^2>/2 using torch."""

    eta_t = torch.tensor(eta, dtype=torch.float64)
    energy = 0.5 * torch.mean(eta_t * eta_t, dim=1)
    eps = torch.tensor(1.0e-20, dtype=torch.float64)

    growth = (torch.log(energy[-1] + eps) - torch.log(energy[0] + eps)) / (2.0 * (t[-1] - t[0]))
    return float(growth.item())


def main() -> None:
    cfg = RTIConfig()
    if cfg.atwood <= 0.0:
        raise AssertionError("Atwood number must be positive for heavy-over-light RTI setup.")

    df = mode_table(cfg)
    unstable_df = df[df["is_unstable"]].copy()
    stable_df = df[~df["is_unstable"]].copy()

    if unstable_df.empty:
        raise AssertionError("No unstable modes found: configuration does not trigger RTI.")
    if stable_df.empty:
        raise AssertionError("No stable modes found: choose parameters with mixed modal behavior.")

    unstable_intervals = find_unstable_intervals(cfg, k_min=1.0e-4, k_max=50.0, grid_size=3600)
    k_critical_formula = np.sqrt(cfg.atwood * cfg.gravity * cfg.rho_sum / cfg.surface_tension)

    dominant_row = unstable_df.loc[unstable_df["growth_rate"].idxmax()]
    dominant_mode = int(dominant_row["mode"])
    dominant_gamma = float(dominant_row["growth_rate"])

    x, t, eta, _ = simulate_interface(df, cfg)

    gamma_fit, fit_r2, fit_mae, amp_trace = estimate_growth_from_fft(eta, t, dominant_mode=dominant_mode)

    _ode_numeric, ode_rel_err = integrate_scalar_growth_ode(
        gamma=dominant_gamma,
        amplitude0=float(amp_trace[0]),
        t=t,
    )

    theoretical_amp = float(amp_trace[0]) * np.exp(dominant_gamma * (t - t[0]))
    rel_amp_end = abs(float(amp_trace[-1]) - float(theoretical_amp[-1])) / max(abs(float(theoretical_amp[-1])), 1.0e-18)

    torch_gamma = torch_energy_growth_rate(eta, t)

    k_critical_numeric = np.nan
    if unstable_intervals:
        k_critical_numeric = unstable_intervals[0][1]

    summary = pd.DataFrame(
        {
            "metric": [
                "atwood_number",
                "unstable_mode_count",
                "stable_mode_count",
                "dominant_mode",
                "dominant_gamma",
                "fft_fitted_gamma",
                "fft_fit_r2",
                "fft_log_mae",
                "ode_rel_error",
                "dominant_mode_end_rel_amp_error",
                "critical_k_formula",
                "critical_k_numeric",
                "torch_energy_gamma",
                "x_grid_size",
                "t_grid_size",
            ],
            "value": [
                cfg.atwood,
                int(unstable_df.shape[0]),
                int(stable_df.shape[0]),
                dominant_mode,
                dominant_gamma,
                gamma_fit,
                fit_r2,
                fit_mae,
                ode_rel_err,
                rel_amp_end,
                float(k_critical_formula),
                float(k_critical_numeric),
                torch_gamma,
                int(x.size),
                int(t.size),
            ],
        }
    )

    print("=== Rayleigh-Taylor Instability MVP (Linear 2-layer Model) ===")
    print(
        "config:",
        {
            "rho_top": cfg.rho_top,
            "rho_bottom": cfg.rho_bottom,
            "atwood": cfg.atwood,
            "gravity": cfg.gravity,
            "surface_tension": cfg.surface_tension,
            "domain_length": cfg.domain_length,
            "max_mode": cfg.max_mode,
            "time_end": cfg.time_end,
        },
    )

    print("\nmode_table:")
    print(df.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    print("\nunstable_k_intervals (from bracketing):")
    if unstable_intervals:
        for left, right in unstable_intervals:
            print(f"  k in ({left:.6f}, {right:.6f})")
    else:
        print("  none")

    print("\nsummary:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    # Acceptance checks for automated validation.
    assert fit_r2 > 0.995, f"Dominant-mode log-amplitude regression too weak: R2={fit_r2:.6f}"
    assert abs(gamma_fit - dominant_gamma) / dominant_gamma < 0.06, (
        "Fitted growth rate deviates from modal prediction by more than 6%."
    )
    assert ode_rel_err < 1.0e-6, f"ODE solver mismatch too large: {ode_rel_err:.3e}"
    assert rel_amp_end < 0.08, f"Dominant mode end-amplitude mismatch too large: {rel_amp_end:.3e}"

    if np.isfinite(k_critical_numeric):
        rel_kc_err = abs(k_critical_numeric - k_critical_formula) / max(k_critical_formula, 1.0e-18)
        assert rel_kc_err < 5.0e-3, f"Critical k mismatch too large: {rel_kc_err:.3e}"

    print("\nPASS: Rayleigh-Taylor linear MVP checks passed.")


if __name__ == "__main__":
    main()
