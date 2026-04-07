"""Minimal runnable MVP for Shapiro delay (gravitational time delay).

The script synthesizes one-way radio propagation delays near solar conjunction,
injects measurement noise, and estimates the PPN parameter gamma via a
transparent least-squares model:

    delay = (1 + gamma) * (G*M/c^3) * ln(4*r1*r2/b^2)

No black-box fitting package is used; each numerical step is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# SI constants
G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
AU = 149_597_870_700.0
R_SUN = 695_700_000.0
EPS = 1e-18


@dataclass(frozen=True)
class DelayConfig:
    """Configuration for a synthetic superior-conjunction ranging campaign."""

    r_earth_m: float = 1.0 * AU
    r_probe_m: float = 1.52 * AU
    lens_mass_kg: float = M_SUN
    true_gamma: float = 1.0

    # Observation campaign around conjunction.
    half_window_days: float = 30.0
    n_observations: int = 121
    alpha_min_deg: float = 0.35
    alpha_slope_deg_per_day: float = 0.16

    # Keep rays outside the solar photosphere.
    b_floor_rsun: float = 1.05

    # Synthetic measurement noise.
    noise_std_s: float = 1.2e-6


def conjunction_geometry(cfg: DelayConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate day offsets, elongation angle, and impact parameter b(t)."""
    days = np.linspace(-cfg.half_window_days, cfg.half_window_days, cfg.n_observations)
    alpha_deg = np.sqrt(cfg.alpha_min_deg**2 + (cfg.alpha_slope_deg_per_day * days) ** 2)
    alpha_rad = np.deg2rad(alpha_deg)

    b = cfg.r_earth_m * np.sin(alpha_rad)
    b_floor_m = cfg.b_floor_rsun * R_SUN
    b = np.maximum(b, b_floor_m)
    return days, alpha_deg, b


def shapiro_log_term(r1_m: float, r2_m: float, b_m: np.ndarray) -> np.ndarray:
    """Return ln(4*r1*r2/b^2), valid in weak-field near-conjunction geometry."""
    argument = (4.0 * r1_m * r2_m) / np.maximum(b_m**2, EPS)
    if np.any(argument <= 1.0):
        raise ValueError("Shapiro log argument must be > 1 for physical delays")
    return np.log(argument)


def shapiro_delay_one_way_s(
    r1_m: float,
    r2_m: float,
    b_m: np.ndarray,
    gamma: float,
    lens_mass_kg: float,
) -> np.ndarray:
    """Compute one-way Shapiro delay in seconds."""
    prefactor = (1.0 + gamma) * (G * lens_mass_kg / C**3)
    return prefactor * shapiro_log_term(r1_m, r2_m, b_m)


def estimate_gamma(delays_s: np.ndarray, basis_s: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Estimate gamma from y = beta*x + noise, where beta=(1+gamma)."""
    denom = float(np.dot(basis_s, basis_s))
    if denom <= 0.0:
        raise ValueError("Degenerate basis for gamma estimation")

    beta_hat = float(np.dot(basis_s, delays_s) / denom)
    fitted = beta_hat * basis_s
    residual = delays_s - fitted

    dof = max(delays_s.size - 1, 1)
    sigma_hat = float(np.sqrt(np.sum(residual**2) / dof))
    se_beta = float(sigma_hat / np.sqrt(denom))

    gamma_hat = beta_hat - 1.0
    se_gamma = se_beta
    return gamma_hat, se_gamma, beta_hat, fitted


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    cfg = DelayConfig()
    rng = np.random.default_rng(20260370)

    days, alpha_deg, b_m = conjunction_geometry(cfg)

    true_delay_s = shapiro_delay_one_way_s(
        r1_m=cfg.r_earth_m,
        r2_m=cfg.r_probe_m,
        b_m=b_m,
        gamma=cfg.true_gamma,
        lens_mass_kg=cfg.lens_mass_kg,
    )

    noise_s = rng.normal(loc=0.0, scale=cfg.noise_std_s, size=true_delay_s.size)
    observed_delay_s = true_delay_s + noise_s

    # Basis corresponds to gamma=0 contribution; model is observed = (1+gamma)*basis + noise.
    basis_s = (G * cfg.lens_mass_kg / C**3) * shapiro_log_term(cfg.r_earth_m, cfg.r_probe_m, b_m)
    gamma_hat, se_gamma, beta_hat, fitted_delay_s = estimate_gamma(observed_delay_s, basis_s)

    gamma0_delay_s = (1.0 + 0.0) * basis_s

    abs_err_gamma = abs(gamma_hat - cfg.true_gamma)
    rmse_fit_us = rmse(observed_delay_s, fitted_delay_s) * 1e6
    rmse_gamma0_us = rmse(observed_delay_s, gamma0_delay_s) * 1e6

    peak_idx = int(np.argmax(true_delay_s))
    edge_avg_us = float(np.mean(np.r_[true_delay_s[:10], true_delay_s[-10:]]) * 1e6)
    peak_delay_us = float(true_delay_s[peak_idx] * 1e6)
    peak_day = float(days[peak_idx])

    ci95_low = gamma_hat - 1.96 * se_gamma
    ci95_high = gamma_hat + 1.96 * se_gamma

    checks = {
        "|gamma_hat - gamma_true| <= 0.06": abs_err_gamma <= 0.06,
        "SE(gamma_hat) <= 0.03": se_gamma <= 0.03,
        "fit RMSE <= 1.25 * noise_std": rmse_fit_us <= 1.25 * (cfg.noise_std_s * 1e6),
        "fit RMSE improves over gamma=0 by >= 40%": rmse_fit_us <= 0.60 * rmse_gamma0_us,
        "conjunction peak delay exceeds edge average by >= 50 us": peak_delay_us >= edge_avg_us + 50.0,
    }

    df = pd.DataFrame(
        {
            "day": days,
            "alpha_deg": alpha_deg,
            "b_over_rsun": b_m / R_SUN,
            "true_delay_us": true_delay_s * 1e6,
            "observed_delay_us": observed_delay_s * 1e6,
            "fitted_delay_us": fitted_delay_s * 1e6,
            "residual_ns": (observed_delay_s - fitted_delay_s) * 1e9,
        }
    )

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("=== Shapiro Delay MVP (PHYS-0370) ===")
    print(
        "geometry: "
        f"r1={cfg.r_earth_m / AU:.2f} AU, r2={cfg.r_probe_m / AU:.2f} AU, "
        f"mass={cfg.lens_mass_kg / M_SUN:.2f} Msun"
    )
    print(
        "campaign: "
        f"N={cfg.n_observations}, days=[{-cfg.half_window_days:.1f}, {cfg.half_window_days:.1f}], "
        f"alpha_min={cfg.alpha_min_deg:.3f} deg, noise_std={cfg.noise_std_s * 1e6:.3f} us"
    )

    print("\nParameter recovery:")
    print(f"beta_hat = {beta_hat:.6f} (target = {1.0 + cfg.true_gamma:.6f})")
    print(f"gamma_hat = {gamma_hat:.6f} +/- {se_gamma:.6f} (1-sigma)")
    print(f"95% CI(gamma): [{ci95_low:.6f}, {ci95_high:.6f}]")
    print(f"absolute error = {abs_err_gamma:.6f}")

    print("\nModel quality:")
    print(f"RMSE(fitted)   = {rmse_fit_us:.6f} us")
    print(f"RMSE(gamma=0)  = {rmse_gamma0_us:.6f} us")
    print(
        f"peak true delay = {peak_delay_us:.3f} us at day {peak_day:.1f}, "
        f"edge average = {edge_avg_us:.3f} us"
    )

    print("\nSample observations (head):")
    print(df.head(8).to_string(index=False))

    print("\nSample observations (around conjunction):")
    mid = cfg.n_observations // 2
    print(df.iloc[mid - 3 : mid + 4].to_string(index=False))

    print("\nSanity checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    all_ok = all(checks.values())
    print(f"\nValidation: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
