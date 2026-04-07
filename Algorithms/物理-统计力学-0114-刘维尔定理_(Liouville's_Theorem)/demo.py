"""MVP demo for Liouville's theorem in statistical mechanics.

The script compares two discretizations of a 1D harmonic oscillator Hamiltonian flow:
- Symplectic Euler (area preserving in phase space)
- Explicit Euler (artificial phase-volume growth)

It prints quantitative diagnostics and performs assertions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OscillatorConfig:
    mass: float
    spring_k: float
    dt: float
    steps: int


@dataclass(frozen=True)
class VolumeReport:
    method: str
    area_initial: float
    area_final: float
    area_ratio: float
    det_per_step: float
    det_total_pred: float


def hamiltonian_vector_field(
    q: np.ndarray, p: np.ndarray, mass: float, spring_k: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dq/dt, dp/dt) for H = p^2/(2m) + k q^2 / 2."""
    dq_dt = p / mass
    dp_dt = -spring_k * q
    return dq_dt, dp_dt


def numerical_phase_divergence(
    q0: float, p0: float, mass: float, spring_k: float, eps: float = 1e-6
) -> float:
    """Numerically estimate div(F) = d(dq/dt)/dq + d(dp/dt)/dp."""
    q_plus = np.array([q0 + eps])
    q_minus = np.array([q0 - eps])
    p_base = np.array([p0])

    dq_plus, _ = hamiltonian_vector_field(q_plus, p_base, mass, spring_k)
    dq_minus, _ = hamiltonian_vector_field(q_minus, p_base, mass, spring_k)
    term_q = (dq_plus[0] - dq_minus[0]) / (2.0 * eps)

    p_plus = np.array([p0 + eps])
    p_minus = np.array([p0 - eps])
    q_base = np.array([q0])

    _, dp_plus = hamiltonian_vector_field(q_base, p_plus, mass, spring_k)
    _, dp_minus = hamiltonian_vector_field(q_base, p_minus, mass, spring_k)
    term_p = (dp_plus[0] - dp_minus[0]) / (2.0 * eps)

    return float(term_q + term_p)


def explicit_euler_step(
    q: np.ndarray, p: np.ndarray, mass: float, spring_k: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    dq_dt, dp_dt = hamiltonian_vector_field(q, p, mass, spring_k)
    q_next = q + dt * dq_dt
    p_next = p + dt * dp_dt
    return q_next, p_next


def symplectic_euler_step(
    q: np.ndarray, p: np.ndarray, mass: float, spring_k: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    # Kick then drift (semi-implicit Euler), symplectic for separable Hamiltonian.
    p_next = p - dt * spring_k * q
    q_next = q + dt * (p_next / mass)
    return q_next, p_next


def covariance_area_proxy(q: np.ndarray, p: np.ndarray) -> float:
    """Return sqrt(det(cov)) as a 2D phase-volume proxy."""
    points = np.column_stack([q, p])
    cov = np.cov(points, rowvar=False)
    det_cov = float(np.linalg.det(cov))
    det_cov = max(det_cov, 0.0)
    return math.sqrt(det_cov)


def one_step_jacobian_det(method: str, cfg: OscillatorConfig) -> float:
    a = cfg.dt / cfg.mass
    b = cfg.dt * cfg.spring_k
    if method == "explicit":
        # Matrix [[1, a],[-b, 1]]
        return 1.0 + a * b
    if method == "symplectic":
        # Matrix [[1-ab, a],[-b, 1]]
        return 1.0
    raise ValueError(f"Unknown method: {method}")


def run_ensemble(
    q0: np.ndarray,
    p0: np.ndarray,
    cfg: OscillatorConfig,
    method: str,
) -> VolumeReport:
    q = q0.copy()
    p = p0.copy()

    area_initial = covariance_area_proxy(q, p)

    for _ in range(cfg.steps):
        if method == "explicit":
            q, p = explicit_euler_step(q, p, cfg.mass, cfg.spring_k, cfg.dt)
        elif method == "symplectic":
            q, p = symplectic_euler_step(q, p, cfg.mass, cfg.spring_k, cfg.dt)
        else:
            raise ValueError(f"Unknown method: {method}")

    area_final = covariance_area_proxy(q, p)
    ratio = area_final / area_initial
    det_step = one_step_jacobian_det(method, cfg)
    det_total_pred = det_step ** cfg.steps

    return VolumeReport(
        method=method,
        area_initial=area_initial,
        area_final=area_final,
        area_ratio=ratio,
        det_per_step=det_step,
        det_total_pred=det_total_pred,
    )


def format_report(report: VolumeReport) -> str:
    return (
        f"[{report.method}] area0={report.area_initial:.6f}, "
        f"areaT={report.area_final:.6f}, "
        f"ratio={report.area_ratio:.6f}, "
        f"det(step)={report.det_per_step:.6f}, "
        f"det(pred,total)={report.det_total_pred:.6f}"
    )


def main() -> None:
    rng = np.random.default_rng(20260407)

    cfg = OscillatorConfig(
        mass=1.0,
        spring_k=1.0,
        dt=0.03,
        steps=1500,
    )

    # Non-isotropic, correlated cloud to avoid degenerate covariance geometry.
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.2, 0.55], [0.55, 0.8]])
    samples = rng.multivariate_normal(mean, cov, size=8000)
    q0 = samples[:, 0]
    p0 = samples[:, 1]

    div_num = numerical_phase_divergence(
        q0=0.37, p0=-0.41, mass=cfg.mass, spring_k=cfg.spring_k
    )

    rep_sym = run_ensemble(q0, p0, cfg, method="symplectic")
    rep_exp = run_ensemble(q0, p0, cfg, method="explicit")

    print("Liouville theorem demo: harmonic oscillator phase flow")
    print(f"config: m={cfg.mass}, k={cfg.spring_k}, dt={cfg.dt}, steps={cfg.steps}")
    print(f"numerical divergence of phase flow ~ {div_num:.3e}")
    print(format_report(rep_sym))
    print(format_report(rep_exp))

    # Validation checks for the MVP.
    assert abs(div_num) < 1e-8, "Hamiltonian phase-flow divergence should be near zero."
    assert abs(rep_sym.area_ratio - 1.0) < 0.03, "Symplectic method should preserve phase volume."
    assert rep_exp.area_ratio > 2.0, "Explicit Euler should show clear phase-volume inflation."

    print("All assertions passed.")


if __name__ == "__main__":
    main()
