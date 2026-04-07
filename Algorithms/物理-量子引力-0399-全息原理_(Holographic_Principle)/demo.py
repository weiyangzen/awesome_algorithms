"""Minimal runnable MVP for the Holographic Principle.

This script builds an auditable numerical chain:
1) geometric radius -> area/volume,
2) black-hole threshold mass -> energy,
3) Bekenstein entropy bound and Bekenstein-Hawking area law in bits,
4) scaling-law regression checks (area ~ R^2, volume ~ R^3),
5) automatic PASS/FAIL validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import constants
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class HolographicConfig:
    """Configuration for deterministic holographic-principle experiments."""

    r_min_m: float = 1.0e-2
    r_max_m: float = 1.0e6
    n_samples: int = 160
    mass_fraction_min: float = 0.05
    mass_fraction_max: float = 0.95
    entropy_density_bits_per_m3: float = 1.0e66
    random_seed: int = 7
    torch_steps: int = 1200
    torch_lr: float = 0.04


def planck_length_squared() -> float:
    """Return l_p^2 = G * hbar / c^3 in SI units."""

    return constants.G * constants.hbar / (constants.c**3)


def sphere_area(radius_m: np.ndarray) -> np.ndarray:
    return 4.0 * np.pi * np.square(radius_m)


def sphere_volume(radius_m: np.ndarray) -> np.ndarray:
    return (4.0 / 3.0) * np.pi * np.power(radius_m, 3.0)


def schwarzschild_mass_from_radius(radius_m: np.ndarray) -> np.ndarray:
    """Mass whose Schwarzschild radius equals `radius_m`."""

    return constants.c**2 * radius_m / (2.0 * constants.G)


def bekenstein_bound_bits(radius_m: np.ndarray, energy_joule: np.ndarray) -> np.ndarray:
    """S <= 2*pi*k_B*E*R/(hbar*c), converted to bits by dividing (k_B*ln 2)."""

    numerator = 2.0 * np.pi * energy_joule * radius_m
    denominator = constants.hbar * constants.c * np.log(2.0)
    return numerator / denominator


def bekenstein_hawking_bits(radius_m: np.ndarray) -> np.ndarray:
    """S_BH = k_B * A / (4 l_p^2), converted to bits."""

    area = sphere_area(radius_m)
    return area / (4.0 * planck_length_squared() * np.log(2.0))


def fit_log_scaling_sklearn(x_log: np.ndarray, y_log: np.ndarray) -> dict[str, float]:
    model = LinearRegression().fit(x_log.reshape(-1, 1), y_log)
    pred = model.predict(x_log.reshape(-1, 1))
    return {
        "method": "sklearn_linear",
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y_log, pred)),
    }


def fit_log_scaling_scipy(x_log: np.ndarray, y_log: np.ndarray) -> dict[str, float]:
    reg = linregress(x_log, y_log)
    return {
        "method": "scipy_linregress",
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "r2": float(reg.rvalue * reg.rvalue),
    }


def fit_log_scaling_torch(
    x_log: np.ndarray,
    y_log: np.ndarray,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)

    x_np = np.asarray(x_log, dtype=float)
    y_np = np.asarray(y_log, dtype=float)
    x_mean = float(np.mean(x_np))
    y_mean = float(np.mean(y_np))
    x_std = float(np.std(x_np))
    y_std = float(np.std(y_np))

    x = torch.tensor((x_np - x_mean) / x_std, dtype=torch.float64)
    y = torch.tensor((y_np - y_mean) / y_std, dtype=torch.float64)

    slope = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    intercept = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.Adam([slope, intercept], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        y_hat = slope * x + intercept
        loss = torch.mean((y_hat - y) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_hat = slope * x + intercept
        ss_res = torch.sum((y - y_hat) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        slope_orig = (y_std / x_std) * float(slope.item())
        intercept_orig = y_std * float(intercept.item()) + y_mean - slope_orig * x_mean

    return {
        "method": "torch_adam",
        "slope": slope_orig,
        "intercept": intercept_orig,
        "r2": float(r2.item()),
    }


def build_dataset(cfg: HolographicConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)

    radius_m = np.geomspace(cfg.r_min_m, cfg.r_max_m, cfg.n_samples)
    area_m2 = sphere_area(radius_m)
    volume_m3 = sphere_volume(radius_m)

    m_schwarzschild = schwarzschild_mass_from_radius(radius_m)
    mass_fraction = rng.uniform(
        low=cfg.mass_fraction_min,
        high=cfg.mass_fraction_max,
        size=cfg.n_samples,
    )
    mass_kg = mass_fraction * m_schwarzschild
    energy_joule = mass_kg * constants.c**2

    bekenstein_bits = bekenstein_bound_bits(radius_m, energy_joule)
    bh_bits = bekenstein_hawking_bits(radius_m)

    toy_volume_bits = cfg.entropy_density_bits_per_m3 * volume_m3

    return pd.DataFrame(
        {
            "radius_m": radius_m,
            "area_m2": area_m2,
            "volume_m3": volume_m3,
            "mass_fraction": mass_fraction,
            "mass_kg": mass_kg,
            "energy_joule": energy_joule,
            "bekenstein_bits": bekenstein_bits,
            "bh_bits": bh_bits,
            "ratio_bekenstein_to_bh": bekenstein_bits / bh_bits,
            "toy_volume_bits": toy_volume_bits,
            "ratio_volume_to_bh": toy_volume_bits / bh_bits,
        }
    )


def run_scaling_regressions(df: pd.DataFrame, cfg: HolographicConfig) -> pd.DataFrame:
    x = np.log10(df["radius_m"].to_numpy(dtype=float))
    y_area = np.log10(df["bh_bits"].to_numpy(dtype=float))
    y_volume = np.log10(df["toy_volume_bits"].to_numpy(dtype=float))

    fits = []
    for family, y in (("area_law_bits", y_area), ("volume_model_bits", y_volume)):
        fits.append({"family": family, **fit_log_scaling_sklearn(x, y)})
        fits.append({"family": family, **fit_log_scaling_scipy(x, y)})
        fits.append(
            {
                "family": family,
                **fit_log_scaling_torch(
                    x_log=x,
                    y_log=y,
                    steps=cfg.torch_steps,
                    lr=cfg.torch_lr,
                    seed=cfg.random_seed,
                ),
            }
        )
    return pd.DataFrame(fits)


def theoretical_crossing_radius(entropy_density_bits_per_m3: float) -> float:
    """Solve rho*4/3*pi*R^3 = alpha*R^2 for R, where alpha is BH bit coefficient."""

    alpha = np.pi * constants.c**3 / (constants.G * constants.hbar * np.log(2.0))
    beta = (4.0 / 3.0) * np.pi * entropy_density_bits_per_m3
    return alpha / beta


def first_numeric_crossing_radius(df: pd.DataFrame) -> float:
    ratio = df["ratio_volume_to_bh"].to_numpy(dtype=float)
    radius = df["radius_m"].to_numpy(dtype=float)
    idx = np.where(ratio >= 1.0)[0]
    if len(idx) == 0:
        return float("nan")
    return float(radius[int(idx[0])])


def main() -> None:
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:.6e}")

    cfg = HolographicConfig()
    df = build_dataset(cfg)
    fit_df = run_scaling_regressions(df, cfg)

    ratio_error = np.abs(df["ratio_bekenstein_to_bh"] - df["mass_fraction"])
    max_ratio_error = float(np.max(ratio_error))

    strict_subcritical = bool(np.all(df["bekenstein_bits"] < df["bh_bits"]))

    area_slopes = fit_df.loc[fit_df["family"] == "area_law_bits", "slope"].to_numpy(dtype=float)
    volume_slopes = fit_df.loc[
        fit_df["family"] == "volume_model_bits", "slope"
    ].to_numpy(dtype=float)
    area_slope_error = float(np.max(np.abs(area_slopes - 2.0)))
    volume_slope_error = float(np.max(np.abs(volume_slopes - 3.0)))

    r_cross_theory = theoretical_crossing_radius(cfg.entropy_density_bits_per_m3)
    r_cross_numeric = first_numeric_crossing_radius(df)
    crossing_ratio = float(r_cross_numeric / r_cross_theory)

    checks = [
        {
            "check": "ratio_bekenstein_to_bh matches mass_fraction",
            "metric": max_ratio_error,
            "threshold": 1e-11,
            "pass": max_ratio_error < 1e-11,
        },
        {
            "check": "subcritical systems stay below BH entropy",
            "metric": float(strict_subcritical),
            "threshold": 1.0,
            "pass": strict_subcritical,
        },
        {
            "check": "area-law slope close to 2",
            "metric": area_slope_error,
            "threshold": 2e-3,
            "pass": area_slope_error < 2e-3,
        },
        {
            "check": "volume-model slope close to 3",
            "metric": volume_slope_error,
            "threshold": 2e-3,
            "pass": volume_slope_error < 2e-3,
        },
        {
            "check": "numeric/theory crossing radius consistency",
            "metric": abs(crossing_ratio - 1.0),
            "threshold": 0.15,
            "pass": abs(crossing_ratio - 1.0) < 0.15,
        },
    ]
    checks_df = pd.DataFrame(checks)
    all_pass = bool(checks_df["pass"].all())

    print("=== Holographic Principle MVP ===")
    print(
        f"n_samples={cfg.n_samples}, radius_range=[{cfg.r_min_m:.3e}, {cfg.r_max_m:.3e}] m, "
        f"mass_fraction_range=[{cfg.mass_fraction_min:.2f}, {cfg.mass_fraction_max:.2f}]"
    )
    print(f"l_p^2 = {planck_length_squared():.6e} m^2")
    print()

    print("[Dataset Preview]")
    preview_cols = [
        "radius_m",
        "mass_fraction",
        "bekenstein_bits",
        "bh_bits",
        "ratio_bekenstein_to_bh",
        "ratio_volume_to_bh",
    ]
    print(df.loc[:, preview_cols].head(8).to_string(index=False))
    print()

    print("[Scaling Regressions on log10(S) = slope*log10(R) + intercept]")
    print(fit_df.to_string(index=False))
    print()

    print("[Crossing Radius]")
    print(
        f"R_cross_theory={r_cross_theory:.6e} m, "
        f"R_cross_numeric={r_cross_numeric:.6e} m, "
        f"numeric/theory={crossing_ratio:.6f}"
    )
    print()

    print("[Validation Checks]")
    print(checks_df.to_string(index=False))
    print(f"\nValidation: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        raise RuntimeError("Validation failed for holographic-principle MVP.")


if __name__ == "__main__":
    main()
