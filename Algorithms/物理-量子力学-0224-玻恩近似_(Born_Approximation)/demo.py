"""Born Approximation MVP.

This script computes first-Born scattering for a central Yukawa potential:
    V(r) = g * exp(-mu * r) / r
It compares a numerical radial integral against the analytic Yukawa formula
and prints a compact validation report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BornConfig:
    mass: float = 1.0
    hbar: float = 1.0
    energy: float = 2.0
    yukawa_g: float = -1.0
    yukawa_mu: float = 1.5
    n_theta: int = 181
    r_min: float = 1e-6
    r_max: float = 40.0
    n_r: int = 20_000
    zero_q_tol: float = 1e-10
    amp_rel_tol: float = 2.0e-2
    dsigma_rel_tol: float = 4.0e-2
    forward_rel_tol: float = 1.0e-3


def wave_number(energy: float, mass: float, hbar: float) -> float:
    if energy <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("energy, mass, and hbar must be positive.")
    return float(np.sqrt(2.0 * mass * energy) / hbar)


def momentum_transfer(k: float, theta: np.ndarray) -> np.ndarray:
    return 2.0 * k * np.sin(theta / 2.0)


def yukawa_potential(r: np.ndarray, g: float, mu: float) -> np.ndarray:
    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    return g * np.exp(-mu * r) / r


def born_amplitude_numeric_central(
    q: np.ndarray,
    r: np.ndarray,
    potential: np.ndarray,
    mass: float,
    hbar: float,
    zero_q_tol: float,
) -> np.ndarray:
    """First Born amplitude for a central potential using radial quadrature.

    f(theta) = -(2m/(hbar^2*q)) * int_0^inf r V(r) sin(qr) dr
    and for q -> 0:
    f(0) = -(2m/hbar^2) * int_0^inf r^2 V(r) dr
    """

    prefactor = -2.0 * mass / (hbar**2)
    radial_kernel = r * potential
    sin_matrix = np.sin(np.outer(q, r))
    integral = np.trapezoid(radial_kernel[None, :] * sin_matrix, x=r, axis=1)

    q_safe = np.where(np.abs(q) < zero_q_tol, 1.0, q)
    amp = prefactor * integral / q_safe

    zero_mask = np.abs(q) < zero_q_tol
    if np.any(zero_mask):
        zero_limit_integral = np.trapezoid((r**2) * potential, x=r)
        amp[zero_mask] = prefactor * zero_limit_integral

    return amp


def born_amplitude_yukawa_analytic(
    q: np.ndarray,
    mass: float,
    hbar: float,
    g: float,
    mu: float,
) -> np.ndarray:
    return -(2.0 * mass * g) / (hbar**2 * (q**2 + mu**2))


def differential_cross_section(amplitude: np.ndarray) -> np.ndarray:
    return np.abs(amplitude) ** 2


def relative_error(numerical: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum(np.abs(reference), eps)
    return np.abs(numerical - reference) / denom


def build_report_table(
    theta_deg: np.ndarray,
    q: np.ndarray,
    f_numeric: np.ndarray,
    f_analytic: np.ndarray,
    dsigma_num: np.ndarray,
    dsigma_analytic: np.ndarray,
) -> pd.DataFrame:
    rel_f = relative_error(f_numeric, f_analytic)
    rel_dsigma = relative_error(dsigma_num, dsigma_analytic)

    return pd.DataFrame(
        {
            "theta_deg": theta_deg,
            "q": q,
            "f_numeric": np.real_if_close(f_numeric),
            "f_analytic": np.real_if_close(f_analytic),
            "rel_err": rel_f,
            "dsigma_num": dsigma_num,
            "dsigma_analytic": dsigma_analytic,
            "dsigma_rel_err": rel_dsigma,
        }
    )


def validate_results(table: pd.DataFrame, cfg: BornConfig) -> tuple[bool, dict[str, float]]:
    max_rel_amp = float(table["rel_err"].max())
    max_rel_dsigma = float(table["dsigma_rel_err"].max())

    zero_idx = int((table["theta_deg"].abs()).idxmin())
    forward_rel_amp = float(table.loc[zero_idx, "rel_err"])
    finite = bool(np.isfinite(table.to_numpy(dtype=float)).all())

    passed = (
        finite
        and max_rel_amp <= cfg.amp_rel_tol
        and max_rel_dsigma <= cfg.dsigma_rel_tol
        and forward_rel_amp <= cfg.forward_rel_tol
    )

    metrics = {
        "max_rel_amp": max_rel_amp,
        "max_rel_dsigma": max_rel_dsigma,
        "forward_rel_amp": forward_rel_amp,
        "finite": float(finite),
    }
    return passed, metrics


def main() -> None:
    cfg = BornConfig()

    theta = np.linspace(0.0, np.pi, cfg.n_theta)
    theta_deg = np.degrees(theta)
    r = np.linspace(cfg.r_min, cfg.r_max, cfg.n_r)

    k = wave_number(cfg.energy, cfg.mass, cfg.hbar)
    q = momentum_transfer(k, theta)
    potential = yukawa_potential(r, cfg.yukawa_g, cfg.yukawa_mu)

    f_numeric = born_amplitude_numeric_central(
        q=q,
        r=r,
        potential=potential,
        mass=cfg.mass,
        hbar=cfg.hbar,
        zero_q_tol=cfg.zero_q_tol,
    )
    f_analytic = born_amplitude_yukawa_analytic(
        q=q,
        mass=cfg.mass,
        hbar=cfg.hbar,
        g=cfg.yukawa_g,
        mu=cfg.yukawa_mu,
    )

    dsigma_num = differential_cross_section(f_numeric)
    dsigma_analytic = differential_cross_section(f_analytic)

    table = build_report_table(theta_deg, q, f_numeric, f_analytic, dsigma_num, dsigma_analytic)
    passed, metrics = validate_results(table, cfg)

    sample_rows = np.linspace(0, len(table) - 1, 10, dtype=int)
    sample = table.iloc[sample_rows]

    print("=== Born Approximation MVP (Yukawa potential) ===")
    print(
        f"m={cfg.mass}, hbar={cfg.hbar}, E={cfg.energy}, "
        f"g={cfg.yukawa_g}, mu={cfg.yukawa_mu}, k={k:.6f}"
    )
    print(f"theta points={cfg.n_theta}, radial points={cfg.n_r}, r_max={cfg.r_max}")
    print()
    print("Sampled angle table:")
    print(sample.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    print("Validation metrics:")
    print(f"max_rel_amp    = {metrics['max_rel_amp']:.6e} (tol={cfg.amp_rel_tol:.2e})")
    print(f"max_rel_dsigma = {metrics['max_rel_dsigma']:.6e} (tol={cfg.dsigma_rel_tol:.2e})")
    print(f"forward_rel_amp= {metrics['forward_rel_amp']:.6e} (tol={cfg.forward_rel_tol:.2e})")
    print(f"finite_check   = {bool(metrics['finite'])}")
    print(f"Validation: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
