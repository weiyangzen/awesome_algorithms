"""AdS/CFT correspondence MVP.

This script demonstrates a minimal, auditable scalar-sector dictionary:
1) bulk mass m^2 in AdS_{d+1} -> boundary scaling dimension Delta,
2) Delta -> CFT two-point power-law correlator,
3) noisy boundary data -> fitted Delta -> reconstructed m^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class AdsCftConfig:
    """Configuration for a deterministic AdS/CFT scalar toy experiment."""

    d: int = 4
    L: float = 1.0
    m2_true: float = -3.0
    amplitude_true: float = 1.8
    r_min: float = 0.35
    r_max: float = 4.0
    n_samples: int = 120
    noise_sigma_log: float = 0.03
    random_seed: int = 7
    torch_steps: int = 1200
    torch_lr: float = 0.05


def bf_bound(d: int) -> float:
    """Breitenlohner-Freedman bound for m^2 L^2 in AdS_{d+1}."""

    return -0.25 * float(d * d)


def delta_branches(d: int, m2_L2: float) -> tuple[float, float]:
    """Return (Delta_-, Delta_+) from m^2 L^2 = Delta(Delta-d)."""

    disc = 0.25 * d * d + m2_L2
    if disc < 0.0:
        raise ValueError("m^2 L^2 violates BF bound; Delta becomes complex.")
    root = float(np.sqrt(disc))
    return 0.5 * d - root, 0.5 * d + root


def mass_from_delta(d: int, delta: float, L: float) -> float:
    """Inverse dictionary: Delta -> m^2."""

    return float(delta * (delta - d) / (L * L))


def cft_two_point(r: np.ndarray, amplitude: float, delta: float) -> np.ndarray:
    """Position-space Euclidean CFT two-point scaling law: A / r^(2 Delta)."""

    return amplitude * np.power(r, -2.0 * delta)


def bulk_to_boundary_kernel(z: np.ndarray, r: float, delta: float, d: int) -> np.ndarray:
    """Scalar bulk-to-boundary propagator K_Delta(z, r) in Euclidean Poincare patch."""

    norm = gamma(delta) / (np.pi ** (0.5 * d) * gamma(delta - 0.5 * d))
    return norm * np.power(z / (z * z + r * r), delta)


def generate_noisy_boundary_data(cfg: AdsCftConfig, delta_true: float) -> pd.DataFrame:
    """Generate synthetic boundary two-point data with multiplicative log-noise."""

    rng = np.random.default_rng(cfg.random_seed)
    r = np.linspace(cfg.r_min, cfg.r_max, cfg.n_samples)
    g_clean = cft_two_point(r, cfg.amplitude_true, delta_true)
    eps = rng.normal(loc=0.0, scale=cfg.noise_sigma_log, size=r.shape)
    g_obs = g_clean * np.exp(eps)
    return pd.DataFrame({"r": r, "G_clean": g_clean, "G_obs": g_obs})


def fit_log_linear(df: pd.DataFrame) -> dict[str, float | np.ndarray | str]:
    """Fit log G = log A - 2 Delta log r using sklearn linear regression."""

    x = np.log(df["r"].to_numpy()).reshape(-1, 1)
    y = np.log(df["G_obs"].to_numpy())
    model = LinearRegression().fit(x, y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    delta = -0.5 * slope
    amplitude = float(np.exp(intercept))

    pred = cft_two_point(df["r"].to_numpy(), amplitude, delta)
    return {
        "method": "sklearn_log_linear",
        "delta": delta,
        "amplitude": amplitude,
        "r2": float(r2_score(df["G_obs"], pred)),
        "mae": float(mean_absolute_error(df["G_obs"], pred)),
        "pred": pred,
    }


def fit_curve_power_law(df: pd.DataFrame) -> dict[str, float | np.ndarray | str]:
    """Fit G(r) = A / r^(2 Delta) directly using scipy curve_fit."""

    r = df["r"].to_numpy()
    y = df["G_obs"].to_numpy()

    popt, _ = curve_fit(
        cft_two_point,
        r,
        y,
        p0=(1.0, 2.0),
        bounds=([1e-12, 0.1], [1e4, 20.0]),
        maxfev=20000,
    )
    amplitude, delta = map(float, popt)

    pred = cft_two_point(r, amplitude, delta)
    return {
        "method": "scipy_curve_fit",
        "delta": delta,
        "amplitude": amplitude,
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "pred": pred,
    }


def fit_torch_log_model(df: pd.DataFrame, cfg: AdsCftConfig) -> dict[str, float | np.ndarray | str]:
    """Fit in log-space with Torch + Adam: y_hat = logA - 2 Delta x."""

    torch.manual_seed(cfg.random_seed)

    x = torch.tensor(np.log(df["r"].to_numpy()), dtype=torch.float64)
    y = torch.tensor(np.log(df["G_obs"].to_numpy()), dtype=torch.float64)

    log_amp = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    delta = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
    optimizer = torch.optim.Adam([log_amp, delta], lr=cfg.torch_lr)

    for _ in range(cfg.torch_steps):
        optimizer.zero_grad()
        y_hat = log_amp - 2.0 * delta * x
        loss = torch.mean((y_hat - y) ** 2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(0.1, 20.0)

    amplitude = float(torch.exp(log_amp).item())
    delta_value = float(delta.item())
    pred = cft_two_point(df["r"].to_numpy(), amplitude, delta_value)

    return {
        "method": "torch_adam_log_fit",
        "delta": delta_value,
        "amplitude": amplitude,
        "r2": float(r2_score(df["G_obs"], pred)),
        "mae": float(mean_absolute_error(df["G_obs"], pred)),
        "pred": pred,
    }


def estimate_near_boundary_slope(delta: float, d: int) -> tuple[float, float]:
    """Check K_Delta(z, r0) ~ z^Delta at fixed r0 and small z."""

    z = np.geomspace(1e-4, 1e-1, 80)
    r0 = 1.2
    k = bulk_to_boundary_kernel(z, r0, delta, d)

    reg = linregress(np.log(z), np.log(k))
    return float(reg.slope), float(reg.rvalue * reg.rvalue)


def summarize_fits(
    fits: list[dict[str, float | np.ndarray | str]], cfg: AdsCftConfig, delta_true: float
) -> pd.DataFrame:
    """Collect all estimators and reconstructed masses in one table."""

    rows: list[dict[str, float | str]] = []
    bound = bf_bound(cfg.d)
    for res in fits:
        delta_est = float(res["delta"])
        m2_est = mass_from_delta(cfg.d, delta_est, cfg.L)
        rows.append(
            {
                "method": str(res["method"]),
                "delta_est": delta_est,
                "delta_abs_err": abs(delta_est - delta_true),
                "m2_est": m2_est,
                "m2_abs_err": abs(m2_est - cfg.m2_true),
                "bf_margin_m2L2": m2_est * cfg.L * cfg.L - bound,
                "r2": float(res["r2"]),
                "mae": float(res["mae"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = AdsCftConfig()

    m2_L2_true = cfg.m2_true * cfg.L * cfg.L
    if m2_L2_true < bf_bound(cfg.d):
        raise ValueError("Configured m2_true violates BF bound.")

    delta_minus, delta_plus = delta_branches(cfg.d, m2_L2_true)
    delta_true = delta_plus  # standard quantization

    df = generate_noisy_boundary_data(cfg, delta_true)

    fit_a = fit_log_linear(df)
    fit_b = fit_curve_power_law(df)
    fit_c = fit_torch_log_model(df, cfg)
    fits = [fit_a, fit_b, fit_c]

    summary = summarize_fits(fits, cfg, delta_true)

    delta_consensus = float(summary["delta_est"].mean())
    m2_consensus = mass_from_delta(cfg.d, delta_consensus, cfg.L)

    slope_z, slope_r2 = estimate_near_boundary_slope(delta_consensus, cfg.d)
    slope_abs_err = abs(slope_z - delta_consensus)

    preview = df.copy()
    preview["G_pred_curve_fit"] = fit_b["pred"]
    preview["abs_residual"] = np.abs(preview["G_obs"] - preview["G_pred_curve_fit"])

    print("=== AdS/CFT Scalar Dictionary MVP ===")
    print(
        f"d={cfg.d}, L={cfg.L:.3f}, m2_true={cfg.m2_true:.6f}, "
        f"Delta_true_plus={delta_true:.6f}, Delta_true_minus={delta_minus:.6f}"
    )
    print(f"BF bound on m^2 L^2: {bf_bound(cfg.d):.6f}")

    print("\n[Estimator Summary]")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n[Consensus Reconstruction]")
    print(
        f"delta_consensus={delta_consensus:.6f}, "
        f"m2_consensus={m2_consensus:.6f}, "
        f"|m2_consensus-m2_true|={abs(m2_consensus-cfg.m2_true):.6f}"
    )

    print("\n[Near-Boundary Check for K_Delta(z, r0)]")
    print(
        f"log-log slope={slope_z:.6f}, slope_r2={slope_r2:.6f}, "
        f"|slope-Delta_consensus|={slope_abs_err:.6f}"
    )

    print("\n[Boundary Data Preview (first 8 rows)]")
    print(
        preview.head(8).to_string(
            index=False,
            float_format=lambda x: f"{x:.6e}",
        )
    )

    checks = {
        "all_fit_delta_error_lt_0.08": bool((summary["delta_abs_err"] < 0.08).all()),
        "all_fit_r2_gt_0.99": bool((summary["r2"] > 0.99).all()),
        "all_fit_mae_lt_0.40": bool((summary["mae"] < 0.40).all()),
        "all_m2_above_bf_bound": bool((summary["bf_margin_m2L2"] >= -1e-10).all()),
        "consensus_m2_error_lt_0.30": bool(abs(m2_consensus - cfg.m2_true) < 0.30),
        "kernel_slope_match_lt_0.04": bool(slope_abs_err < 0.04),
    }

    checks_df = pd.DataFrame(
        {
            "check": list(checks.keys()),
            "passed": list(checks.values()),
        }
    )
    print("\n[Validation Checks]")
    print(checks_df.to_string(index=False))

    passed = bool(all(checks.values()))
    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
