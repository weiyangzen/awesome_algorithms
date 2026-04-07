"""Cutoff Regularization MVP.

This script demonstrates one-loop cutoff regularization on

    I(Lambda, m) = ∫_{|k|<Lambda} d^4k/(2π)^4 * 1/(k^2 + m^2)^2

in Euclidean space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate


@dataclass(frozen=True)
class CutoffConfig:
    """Configuration for the cutoff-regularization demo."""

    mass: float = 0.7
    mu_ren: float = 1.0
    cutoffs: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
    epsabs: float = 1e-11
    epsrel: float = 1e-10


def validate_config(cfg: CutoffConfig) -> None:
    """Validate basic physical and numerical constraints."""
    if cfg.mass <= 0.0:
        raise ValueError("mass must be > 0.")
    if cfg.mu_ren <= 0.0:
        raise ValueError("mu_ren must be > 0.")
    if len(cfg.cutoffs) == 0:
        raise ValueError("cutoffs must be non-empty.")
    if any(c <= 0.0 for c in cfg.cutoffs):
        raise ValueError("all cutoff values must be > 0.")
    if any(cfg.cutoffs[i] >= cfg.cutoffs[i + 1] for i in range(len(cfg.cutoffs) - 1)):
        raise ValueError("cutoffs must be strictly increasing.")


def cutoff_integrand_radial(k: float, mass: float) -> float:
    """Radial integrand in 4D: k^3/(k^2 + m^2)^2."""
    return (k**3) / ((k * k + mass * mass) ** 2)


def one_loop_cutoff_numeric(lambda_uv: float, mass: float, epsabs: float, epsrel: float) -> float:
    """Numerically evaluate the cutoff integral by radial quadrature."""
    integral, err = integrate.quad(
        cutoff_integrand_radial,
        0.0,
        lambda_uv,
        args=(mass,),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
    )
    if (not np.isfinite(integral)) or (err > 1e-8):
        raise RuntimeError(f"Unexpected integration issue: integral={integral}, err={err}")
    return integral / (8.0 * np.pi * np.pi)


def one_loop_cutoff_analytic(lambda_uv: float, mass: float) -> float:
    """Closed-form expression for the same cutoff-regularized integral."""
    r = (lambda_uv * lambda_uv) / (mass * mass)
    return (math.log(1.0 + r) - (r / (1.0 + r))) / (16.0 * np.pi * np.pi)


def one_loop_cutoff_asymptotic(lambda_uv: float, mass: float) -> float:
    """Large-cutoff asymptotic expansion up to O(m^2/Lambda^2)."""
    lam2 = lambda_uv * lambda_uv
    m2 = mass * mass
    return (math.log(lam2 / m2) - 1.0 + (2.0 * m2 / lam2)) / (16.0 * np.pi * np.pi)


def counterterm_log(lambda_uv: float, mu_ren: float) -> float:
    """A simple logarithmic subtraction counterterm."""
    return (math.log((lambda_uv * lambda_uv) / (mu_ren * mu_ren)) - 1.0) / (16.0 * np.pi * np.pi)


def renormalized_subtracted(lambda_uv: float, mass: float, mu_ren: float) -> float:
    """Finite quantity after explicit cutoff subtraction."""
    return one_loop_cutoff_analytic(lambda_uv, mass) - counterterm_log(lambda_uv, mu_ren)


def renormalized_limit(mass: float, mu_ren: float) -> float:
    """Lambda -> infinity limit of the subtracted quantity."""
    return math.log((mu_ren * mu_ren) / (mass * mass)) / (16.0 * np.pi * np.pi)


def build_report(cfg: CutoffConfig) -> pd.DataFrame:
    """Build a structured table of exact/numeric/asymptotic/subtracted values."""
    rows: list[dict[str, float]] = []
    target_limit = renormalized_limit(cfg.mass, cfg.mu_ren)

    for lam in cfg.cutoffs:
        i_num = one_loop_cutoff_numeric(lam, cfg.mass, cfg.epsabs, cfg.epsrel)
        i_exact = one_loop_cutoff_analytic(lam, cfg.mass)
        i_asym = one_loop_cutoff_asymptotic(lam, cfg.mass)
        i_ren = renormalized_subtracted(lam, cfg.mass, cfg.mu_ren)

        rows.append(
            {
                "Lambda": lam,
                "I_numeric": i_num,
                "I_analytic": i_exact,
                "I_asymptotic": i_asym,
                "I_ren_subtracted": i_ren,
                "abs_err_num_vs_analytic": abs(i_num - i_exact),
                "abs_err_asymptotic": abs(i_asym - i_exact),
                "abs_err_ren_to_limit": abs(i_ren - target_limit),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = CutoffConfig()
    validate_config(cfg)

    df = build_report(cfg)
    limit_val = renormalized_limit(cfg.mass, cfg.mu_ren)

    pd.set_option("display.float_format", lambda x: f"{x:.12e}")
    print("=== Cutoff Regularization: one-loop scalar integral ===")
    print(f"mass = {cfg.mass:.6f}, mu_ren = {cfg.mu_ren:.6f}")
    print(f"renormalized limit (Lambda->inf) = {limit_val:.12e}\n")
    print(df.to_string(index=False))

    max_num_err = float(df["abs_err_num_vs_analytic"].max())
    max_ren_err = float(df["abs_err_ren_to_limit"].iloc[-1])
    print(f"\nmax numeric-vs-analytic error = {max_num_err:.3e}")
    print(f"largest-cutoff renormalized error = {max_ren_err:.3e}")

    # 1) Numeric quadrature must match the exact closed form.
    assert max_num_err < 1e-10, "numeric and analytic results mismatch."

    # 2) Bare integral should increase with cutoff (log divergence).
    analytic_vals = df["I_analytic"].to_numpy()
    assert np.all(np.diff(analytic_vals) > 0.0), "bare integral is not monotonically increasing."

    # 3) Asymptotic approximation should improve as Lambda grows.
    asym_err = df["abs_err_asymptotic"].to_numpy()
    assert np.all(np.diff(asym_err[2:]) < 0.0), "asymptotic error does not decrease at large cutoff."

    # 4) Subtracted result should approach finite limit.
    assert max_ren_err < 1e-6, "renormalized quantity did not approach expected finite limit."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
