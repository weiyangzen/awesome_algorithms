"""Minimal runnable MVP for LO non-singlet DGLAP evolution in x-space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvolutionConfig:
    """Numerical and physical parameters for the MVP evolution."""

    n_f: int = 4
    lambda_qcd_gev: float = 0.20
    q0_gev: float = 2.0
    q1_gev: float = 20.0
    x_min: float = 1e-3
    x_max: float = 0.90
    n_x: int = 64
    n_t: int = 72
    n_z: int = 180
    a_init: float = 0.50
    b_init: float = 3.00


def alpha_s_lo(q2_gev2: float, n_f: int, lambda_qcd_gev: float) -> float:
    """One-loop running coupling alpha_s(Q^2)."""
    if q2_gev2 <= 0.0:
        raise ValueError("q2_gev2 must be positive.")
    if lambda_qcd_gev <= 0.0:
        raise ValueError("lambda_qcd_gev must be positive.")
    beta0 = 11.0 - (2.0 / 3.0) * n_f
    if beta0 <= 0.0:
        raise ValueError("beta0 must be positive for this LO asymptotically-free setup.")

    lambda2 = lambda_qcd_gev**2
    if q2_gev2 <= lambda2:
        raise ValueError("Q^2 must stay above Lambda_QCD^2 in this perturbative MVP.")

    return 4.0 * np.pi / (beta0 * np.log(q2_gev2 / lambda2))


def build_x_grid(x_min: float, x_max: float, n_x: int) -> np.ndarray:
    """Build a geometric x-grid in (0, 1)."""
    if not (0.0 < x_min < x_max < 1.0):
        raise ValueError("Require 0 < x_min < x_max < 1.")
    if n_x < 8:
        raise ValueError("n_x is too small for a stable toy evolution.")
    return np.geomspace(x_min, x_max, n_x)


def initial_non_singlet_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Simple analytic initial condition q(x,Q0^2) = x^a (1-x)^b."""
    if np.any(x <= 0.0) or np.any(x >= 1.0):
        raise ValueError("x must be strictly inside (0, 1).")
    if a <= -1.0 or b <= -1.0:
        raise ValueError("a and b should keep the profile integrable in (0,1).")
    return np.power(x, a) * np.power(1.0 - x, b)


def plus_subtraction_integral_0_to_x(x: float) -> float:
    """Return integral_0^x dz/(1-z), used by the plus-distribution subtraction."""
    if not (0.0 <= x < 1.0):
        raise ValueError("x must satisfy 0 <= x < 1.")
    return -np.log(1.0 - x)


def non_singlet_convolution_lo(x_grid: np.ndarray, q_vals: np.ndarray, n_z: int) -> np.ndarray:
    """Compute (P_qq ⊗ q)(x) at LO with explicit plus-prescription handling."""
    if x_grid.shape != q_vals.shape:
        raise ValueError("x_grid and q_vals must share the same shape.")
    if n_z < 32:
        raise ValueError("n_z too small for stable convolution integral.")

    c_f = 4.0 / 3.0
    eps = 1e-8
    conv = np.empty_like(q_vals)

    for i, x in enumerate(x_grid):
        z = np.linspace(x, 1.0 - eps, n_z)
        y = x / z
        phi = np.interp(y, x_grid, q_vals, left=q_vals[0], right=q_vals[-1]) / z
        phi_at_1 = q_vals[i]

        # Decompose P_qq(z) as: 2/(1-z)_+ - (1+z) + (3/2) delta(1-z).
        singular_plus = (
            2.0 * np.trapezoid((phi - phi_at_1) / (1.0 - z), z)
            - 2.0 * phi_at_1 * plus_subtraction_integral_0_to_x(float(x))
        )
        regular_part = -np.trapezoid((1.0 + z) * phi, z)
        delta_term = 1.5 * phi_at_1

        conv[i] = c_f * (singular_plus + regular_part + delta_term)

    return conv


def dglap_rhs_lo(
    x_grid: np.ndarray,
    q_vals: np.ndarray,
    q2_gev2: float,
    n_f: int,
    lambda_qcd_gev: float,
    n_z: int,
) -> np.ndarray:
    """RHS: dq/dlnQ^2 = alpha_s/(2pi) * (P_qq ⊗ q)."""
    alpha = alpha_s_lo(q2_gev2=q2_gev2, n_f=n_f, lambda_qcd_gev=lambda_qcd_gev)
    conv = non_singlet_convolution_lo(x_grid=x_grid, q_vals=q_vals, n_z=n_z)
    return (alpha / (2.0 * np.pi)) * conv


def evolve_pdf_euler(config: EvolutionConfig, x_grid: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evolve q(x,Q^2) from Q0 to Q1 using explicit Euler in t=ln(Q^2)."""
    t0 = np.log(config.q0_gev**2)
    t1 = np.log(config.q1_gev**2)
    if t1 <= t0:
        raise ValueError("Need Q1 > Q0 for forward evolution.")

    dt = (t1 - t0) / config.n_t
    t_path = np.linspace(t0, t1, config.n_t + 1)

    q = q0.copy()
    for step in range(config.n_t):
        q2 = float(np.exp(t_path[step]))
        dq_dt = dglap_rhs_lo(
            x_grid=x_grid,
            q_vals=q,
            q2_gev2=q2,
            n_f=config.n_f,
            lambda_qcd_gev=config.lambda_qcd_gev,
            n_z=config.n_z,
        )
        q = q + dt * dq_dt
        q = np.maximum(q, 1e-12)

    return q, t_path


def summarize_selected_points(x_grid: np.ndarray, q_initial: np.ndarray, q_final: np.ndarray) -> pd.DataFrame:
    """Create a compact table at representative x points."""
    probe_x = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 6e-1, 8.5e-1], dtype=float)
    rows: list[dict[str, float]] = []
    for x in probe_x:
        qi = float(np.interp(x, x_grid, q_initial))
        qf = float(np.interp(x, x_grid, q_final))
        rows.append(
            {
                "x": x,
                "q_initial": qi,
                "q_final": qf,
                "ratio_qf_over_qi": qf / max(qi, 1e-15),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = EvolutionConfig()

    x_grid = build_x_grid(x_min=config.x_min, x_max=config.x_max, n_x=config.n_x)
    q_initial = initial_non_singlet_pdf(x=x_grid, a=config.a_init, b=config.b_init)

    q_final, _ = evolve_pdf_euler(config=config, x_grid=x_grid, q0=q_initial)

    q0_int = float(np.trapezoid(q_initial, x_grid))
    q1_int = float(np.trapezoid(q_final, x_grid))
    rel_drift = abs(q1_int - q0_int) / max(abs(q0_int), 1e-15)

    alpha_q0 = alpha_s_lo(config.q0_gev**2, config.n_f, config.lambda_qcd_gev)
    alpha_q1 = alpha_s_lo(config.q1_gev**2, config.n_f, config.lambda_qcd_gev)

    report_df = summarize_selected_points(x_grid=x_grid, q_initial=q_initial, q_final=q_final)

    x_high = 0.6
    q_high_initial = float(np.interp(x_high, x_grid, q_initial))
    q_high_final = float(np.interp(x_high, x_grid, q_final))

    print("=== LO Non-singlet DGLAP MVP ===")
    print(
        "Config: "
        f"N_f={config.n_f}, Lambda_QCD={config.lambda_qcd_gev:.3f} GeV, "
        f"Q0={config.q0_gev:.2f} GeV, Q1={config.q1_gev:.2f} GeV, "
        f"N_x={config.n_x}, N_t={config.n_t}, N_z={config.n_z}"
    )
    print(f"alpha_s(Q0^2) = {alpha_q0:.6f}")
    print(f"alpha_s(Q1^2) = {alpha_q1:.6f}")
    print()
    print("=== PDF Evolution at Representative x ===")
    print(report_df.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print()
    print("=== Integral Check (non-singlet normalization) ===")
    print(f"Integral at Q0: {q0_int:.8f}")
    print(f"Integral at Q1: {q1_int:.8f}")
    print(f"Relative drift: {rel_drift:.6%}")

    # Numerical/physical sanity checks for this MVP.
    assert np.isfinite(q_final).all() and np.all(q_final > 0.0), "Evolved PDF must stay finite and positive."
    assert alpha_q1 < alpha_q0, "alpha_s should decrease from Q0 to Q1 in asymptotically free regime."
    assert rel_drift < 0.08, f"Integral drift too large for this discretization: {rel_drift:.3%}"
    assert q_high_final < q_high_initial, "High-x tail should be suppressed after evolution."
    assert np.max(np.abs(q_final - q_initial)) > 1e-4, "Evolution effect is unexpectedly tiny."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
