"""Minimal runnable MVP for Higgs Mechanism in an Abelian toy model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HiggsConfig:
    """Numerical configuration for the Abelian Higgs toy model."""

    mu: float = 1.7
    lambda_: float = 0.9
    gauge_coupling: float = 0.65
    learning_rate: float = 0.05
    max_steps: int = 4000
    grad_tol: float = 1e-8
    n_trials: int = 6
    random_seed: int = 67
    ring_scan_points: int = 2001


def potential(phi: np.ndarray, cfg: HiggsConfig) -> np.ndarray:
    """Mexican-hat potential: V = -mu^2/2 * |phi|^2 + lambda/4 * |phi|^4."""
    radius_sq = np.sum(phi * phi, axis=-1)
    return -0.5 * cfg.mu**2 * radius_sq + 0.25 * cfg.lambda_ * radius_sq**2


def gradient(phi: np.ndarray, cfg: HiggsConfig) -> np.ndarray:
    """Gradient of the potential with respect to the two real scalar components."""
    radius_sq = float(np.dot(phi, phi))
    return (-cfg.mu**2 + cfg.lambda_ * radius_sq) * phi


def hessian(phi: np.ndarray, cfg: HiggsConfig) -> np.ndarray:
    """2x2 Hessian matrix d^2V / dphi_i dphi_j."""
    x, y = float(phi[0]), float(phi[1])
    radius_sq = x * x + y * y
    common = -cfg.mu**2 + cfg.lambda_ * radius_sq
    return np.array(
        [
            [common + 2.0 * cfg.lambda_ * x * x, 2.0 * cfg.lambda_ * x * y],
            [2.0 * cfg.lambda_ * x * y, common + 2.0 * cfg.lambda_ * y * y],
        ],
        dtype=float,
    )


def rotate_to_unitary_gauge(phi: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate by a global U(1) phase so that phi_2 -> 0 (unitary gauge direction)."""
    theta = float(np.arctan2(phi[1], phi[0]))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rotated = np.array([phi[0] * c + phi[1] * s, -phi[0] * s + phi[1] * c], dtype=float)
    return rotated, theta


def gradient_flow(initial_phi: np.ndarray, cfg: HiggsConfig) -> tuple[np.ndarray, int, float]:
    """Simple explicit gradient flow: phi <- phi - lr * grad(V)."""
    phi = np.array(initial_phi, dtype=float)
    grad_norm = np.inf

    for step in range(1, cfg.max_steps + 1):
        grad = gradient(phi, cfg)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < cfg.grad_tol:
            return phi, step, grad_norm
        phi = phi - cfg.learning_rate * grad

    return phi, cfg.max_steps, grad_norm


def safe_relative_error(numerical: float, analytic: float, eps: float = 1e-12) -> float:
    """Relative error with a stable denominator for near-zero analytic values."""
    return abs(numerical - analytic) / max(abs(analytic), eps)


def main() -> None:
    cfg = HiggsConfig()

    v_analytic = cfg.mu / np.sqrt(cfg.lambda_)
    m_h2_analytic = 2.0 * cfg.mu**2
    m_goldstone2_analytic = 0.0
    m_a2_analytic = (cfg.gauge_coupling * v_analytic) ** 2

    rng = np.random.default_rng(cfg.random_seed)
    initials = rng.normal(loc=0.0, scale=1.5, size=(cfg.n_trials, 2))

    trial_rows: list[dict[str, float | int]] = []
    trial_finals: list[np.ndarray] = []

    for idx, initial_phi in enumerate(initials, start=1):
        final_phi, steps, grad_norm = gradient_flow(initial_phi, cfg)
        trial_finals.append(final_phi)
        radius = float(np.linalg.norm(final_phi))
        phase_deg = float(np.degrees(np.arctan2(final_phi[1], final_phi[0])))
        trial_rows.append(
            {
                "trial": idx,
                "phi1_init": float(initial_phi[0]),
                "phi2_init": float(initial_phi[1]),
                "phi1_final": float(final_phi[0]),
                "phi2_final": float(final_phi[1]),
                "radius_final": radius,
                "phase_deg": phase_deg,
                "steps": steps,
                "grad_norm": grad_norm,
                "V_final": float(potential(final_phi[np.newaxis, :], cfg)[0]),
            }
        )

    trial_df = pd.DataFrame(trial_rows)

    best_idx = int(np.argmin(trial_df["V_final"].to_numpy()))
    selected_vacuum = trial_finals[best_idx]
    selected_radius = float(np.linalg.norm(selected_vacuum))

    unitary_vacuum, theta = rotate_to_unitary_gauge(selected_vacuum)
    unitary_vacuum[0] = abs(unitary_vacuum[0])

    mass_matrix = hessian(unitary_vacuum, cfg)
    eigvals = np.sort(np.linalg.eigvalsh(mass_matrix))
    m_goldstone2_num = float(eigvals[0])
    m_h2_num = float(eigvals[1])
    m_a2_num = (cfg.gauge_coupling * float(unitary_vacuum[0])) ** 2

    ring_scan_r = np.linspace(0.0, 2.0 * v_analytic, cfg.ring_scan_points)
    ring_scan_phi = np.column_stack((ring_scan_r, np.zeros_like(ring_scan_r)))
    ring_scan_v = potential(ring_scan_phi, cfg)
    r_min_scan = float(ring_scan_r[int(np.argmin(ring_scan_v))])

    mass_df = pd.DataFrame(
        [
            {
                "quantity": "v (VEV)",
                "analytic": v_analytic,
                "numerical": float(unitary_vacuum[0]),
            },
            {
                "quantity": "m_H^2",
                "analytic": m_h2_analytic,
                "numerical": m_h2_num,
            },
            {
                "quantity": "m_G^2",
                "analytic": m_goldstone2_analytic,
                "numerical": m_goldstone2_num,
            },
            {
                "quantity": "m_A^2",
                "analytic": m_a2_analytic,
                "numerical": m_a2_num,
            },
        ]
    )
    mass_df["abs_error"] = (mass_df["numerical"] - mass_df["analytic"]).abs()
    mass_df["rel_error"] = [
        np.nan if abs(float(a)) < 1e-12 else safe_relative_error(float(n), float(a))
        for n, a in zip(mass_df["numerical"], mass_df["analytic"])
    ]

    radius_errors = np.abs(trial_df["radius_final"].to_numpy() - v_analytic)

    checks = [
        (
            "ring_scan_min_close",
            abs(r_min_scan - v_analytic) < 3e-3,
            f"|r_min_scan-v|={abs(r_min_scan - v_analytic):.3e}",
        ),
        (
            "selected_vacuum_radius_close",
            abs(selected_radius - v_analytic) < 1e-2,
            f"|r_selected-v|={abs(selected_radius - v_analytic):.3e}",
        ),
        (
            "all_trials_converge_to_ring",
            float(radius_errors.max()) < 4e-2,
            f"max|r_i-v|={float(radius_errors.max()):.3e}",
        ),
        (
            "goldstone_mass_near_zero",
            abs(m_goldstone2_num) < 1e-3,
            f"|m_G^2|={abs(m_goldstone2_num):.3e}",
        ),
        (
            "higgs_mass_matches_theory",
            safe_relative_error(m_h2_num, m_h2_analytic) < 1e-2,
            f"rel_err(m_H^2)={safe_relative_error(m_h2_num, m_h2_analytic):.3e}",
        ),
        (
            "gauge_mass_matches_theory",
            safe_relative_error(m_a2_num, m_a2_analytic) < 1e-2,
            f"rel_err(m_A^2)={safe_relative_error(m_a2_num, m_a2_analytic):.3e}",
        ),
        (
            "higgs_mode_positive",
            m_h2_num > 0.0,
            f"m_H^2={m_h2_num:.6f}",
        ),
    ]

    print("=== Abelian Higgs Mechanism MVP ===")
    print(
        "params:",
        {
            "mu": cfg.mu,
            "lambda": cfg.lambda_,
            "g": cfg.gauge_coupling,
            "lr": cfg.learning_rate,
            "trials": cfg.n_trials,
        },
    )

    print("\n=== Gradient-flow vacuum selection ===")
    print(
        trial_df.to_string(
            index=False,
            float_format=lambda x: f"{x: .6f}",
        )
    )

    print("\n=== Gauge fixing and mass extraction ===")
    print(
        f"selected trial = {best_idx + 1}, raw vacuum = {selected_vacuum}, "
        f"phase = {np.degrees(theta):.3f} deg"
    )
    print(f"unitary-gauge vacuum = {unitary_vacuum}")
    print(f"Hessian eigenvalues [m_G^2, m_H^2] = {eigvals}")
    print(f"ring-scan minimum radius = {r_min_scan:.6f}, analytic v = {v_analytic:.6f}")

    print("\n=== Analytic vs Numerical ===")
    print(
        mass_df.to_string(
            index=False,
            float_format=lambda x: f"{x: .6e}",
        )
    )

    print("\n=== Validation Checks ===")
    all_pass = True
    for name, passed, detail in checks:
        state = "PASS" if passed else "FAIL"
        print(f"[{state}] {name}: {detail}")
        all_pass = all_pass and passed

    if all_pass:
        print("\nValidation: PASS")
        return

    print("\nValidation: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
