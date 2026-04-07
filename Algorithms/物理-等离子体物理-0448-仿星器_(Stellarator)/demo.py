"""Minimal stellarator field-line MVP with diagnostics and validation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
import torch


@dataclass(frozen=True)
class StellaratorConfig:
    """Configuration for a reduced Hamiltonian stellarator model."""

    n_field_periods: int = 5
    m_pol: int = 6
    n_tor: int = 1
    iota0: float = 0.36
    iota1: float = 0.28
    helical_perturbation: float = 0.045
    n_lines: int = 9
    n_toroidal_turns: int = 90
    r0_min: float = 0.18
    r0_max: float = 0.78
    edge_threshold: float = 0.95


def validate_config(cfg: StellaratorConfig) -> None:
    if cfg.n_field_periods <= 0:
        raise ValueError("n_field_periods must be positive")
    if cfg.m_pol <= 0 or cfg.n_tor <= 0:
        raise ValueError("m_pol and n_tor must be positive")
    if cfg.n_lines < 3:
        raise ValueError("n_lines must be >= 3")
    if cfg.n_toroidal_turns <= 5:
        raise ValueError("n_toroidal_turns must be > 5")
    if not (0.0 < cfg.r0_min < cfg.r0_max < 1.0):
        raise ValueError("Require 0 < r0_min < r0_max < 1")
    if not (0.0 < cfg.helical_perturbation < 0.5):
        raise ValueError("helical_perturbation should be in (0, 0.5)")
    if not (0.8 <= cfg.edge_threshold <= 1.2):
        raise ValueError("edge_threshold should be in [0.8, 1.2]")


def field_line_rhs(phi: float, y: np.ndarray, cfg: StellaratorConfig) -> np.ndarray:
    """Reduced field-line dynamics in (r, theta) with phi as independent variable.

    Model from a simple non-axisymmetric Hamiltonian:
        H(r, theta, phi) = iota0*r^2/2 + iota1*r^4/4
                            + epsilon*r^2*cos(m*theta - n*Nfp*phi)/2
    leading to
        dr/dphi = 0.5*epsilon*m*r*sin(phase)
        dtheta/dphi = iota0 + iota1*r^2 + epsilon*cos(phase)
    """

    r, theta = y
    phase = cfg.m_pol * theta - cfg.n_tor * cfg.n_field_periods * phi
    dr_dphi = 0.5 * cfg.helical_perturbation * cfg.m_pol * r * math.sin(phase)
    dtheta_dphi = cfg.iota0 + cfg.iota1 * r * r + cfg.helical_perturbation * math.cos(phase)
    return np.array([dr_dphi, dtheta_dphi], dtype=float)


def estimate_effective_iota(phi: np.ndarray, theta: np.ndarray) -> tuple[float, float]:
    """Fit theta(phi) slope as effective rotational transform and return (iota, R^2)."""

    phi_2d = phi.reshape(-1, 1)
    # `solve_ivp` keeps theta continuous already; avoid `np.unwrap` here because
    # sparse toroidal sampling can jump by >pi and trigger incorrect 2pi branch shifts.
    theta_unwrapped = theta
    reg = LinearRegression()
    reg.fit(phi_2d, theta_unwrapped)
    iota_eff = float(reg.coef_[0])
    r2 = float(reg.score(phi_2d, theta_unwrapped))
    return iota_eff, r2


def trace_single_line(
    cfg: StellaratorConfig,
    line_id: int,
    r0: float,
    theta0: float,
) -> tuple[pd.DataFrame, dict[str, float | bool | int]]:
    """Integrate one field line and return turn-by-turn section points + summary."""

    phi_end = 2.0 * math.pi * cfg.n_toroidal_turns
    phi_sections = np.linspace(0.0, phi_end, cfg.n_toroidal_turns + 1)

    sol = solve_ivp(
        fun=lambda phi, y: field_line_rhs(phi, y, cfg),
        t_span=(0.0, phi_end),
        y0=np.array([r0, theta0], dtype=float),
        method="RK45",
        t_eval=phi_sections,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.35,
    )
    if not sol.success:
        raise RuntimeError(f"Field-line integration failed for line {line_id}: {sol.message}")

    r = sol.y[0]
    theta = sol.y[1]
    theta_mod = np.mod(theta, 2.0 * math.pi)

    iota_eff, fit_r2 = estimate_effective_iota(sol.t, theta)

    section_df = pd.DataFrame(
        {
            "line_id": line_id,
            "turn": np.arange(sol.t.size),
            "phi": sol.t,
            "r": r,
            "theta_mod": theta_mod,
        }
    )

    summary = {
        "line_id": line_id,
        "r0": float(r0),
        "theta0": float(theta0),
        "r_mean": float(np.mean(r)),
        "r_std": float(np.std(r)),
        "r_span": float(np.max(r) - np.min(r)),
        "r_min": float(np.min(r)),
        "r_max": float(np.max(r)),
        "edge_crossed": bool(np.any(r >= cfg.edge_threshold)),
        "iota_eff": iota_eff,
        "iota_fit_r2": fit_r2,
    }
    return section_df, summary


def torch_confinement_sensitivity(
    cfg: StellaratorConfig,
    r0: np.ndarray,
    theta0: np.ndarray,
) -> dict[str, float]:
    """Use PyTorch autograd to estimate d(loss)/d(epsilon) for a surrogate map.

    This is a lightweight differentiable Euler rollout of the same reduced model.
    """

    dtype = torch.float64
    device = "cpu"

    r_init = torch.tensor(r0, dtype=dtype, device=device)
    th = torch.tensor(theta0, dtype=dtype, device=device)
    r = r_init.clone()

    epsilon = torch.tensor(cfg.helical_perturbation, dtype=dtype, device=device, requires_grad=True)

    n_steps = 600
    phi_end = 2.0 * math.pi * cfg.n_toroidal_turns
    dphi = phi_end / n_steps

    phi_val = 0.0
    for _ in range(n_steps):
        phase = cfg.m_pol * th - cfg.n_tor * cfg.n_field_periods * phi_val
        dr = 0.5 * epsilon * cfg.m_pol * r * torch.sin(phase)
        dth = cfg.iota0 + cfg.iota1 * r * r + epsilon * torch.cos(phase)
        r = r + dphi * dr
        th = th + dphi * dth
        phi_val += dphi

    loss = torch.mean((r - r_init) ** 2)
    loss.backward()
    grad = float(epsilon.grad.item())

    lr = 0.10
    with torch.no_grad():
        eps_new = torch.clamp(epsilon - lr * epsilon.grad, min=1e-4, max=0.35)

    return {
        "torch_loss": float(loss.item()),
        "dloss_depsilon": grad,
        "epsilon_suggested": float(eps_new.item()),
    }


def run_experiment(cfg: StellaratorConfig) -> dict[str, pd.DataFrame | float | int]:
    validate_config(cfg)

    r0_values = np.linspace(cfg.r0_min, cfg.r0_max, cfg.n_lines)
    theta0_values = np.linspace(0.0, 2.0 * math.pi, cfg.n_lines, endpoint=False)

    section_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | bool | int]] = []

    for line_id, (r0, theta0) in enumerate(zip(r0_values, theta0_values), start=1):
        section_df, summary = trace_single_line(cfg=cfg, line_id=line_id, r0=float(r0), theta0=float(theta0))
        section_frames.append(section_df)
        summaries.append(summary)

    sections = pd.concat(section_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("line_id").reset_index(drop=True)

    confined_fraction = float((~summary_df["edge_crossed"]).mean())
    avg_iota = float(summary_df["iota_eff"].mean())
    avg_span = float(summary_df["r_span"].mean())

    torch_diag = torch_confinement_sensitivity(cfg=cfg, r0=r0_values, theta0=theta0_values)

    return {
        "sections": sections,
        "summary": summary_df,
        "confined_fraction": confined_fraction,
        "avg_iota": avg_iota,
        "avg_span": avg_span,
        "torch_loss": torch_diag["torch_loss"],
        "torch_grad": torch_diag["dloss_depsilon"],
        "eps_suggested": torch_diag["epsilon_suggested"],
    }


def main() -> None:
    cfg = StellaratorConfig()
    out = run_experiment(cfg)

    summary = out["summary"]
    if not isinstance(summary, pd.DataFrame):
        raise RuntimeError("summary table is missing")

    preview = summary[["line_id", "r0", "r_mean", "r_span", "r_max", "iota_eff", "edge_crossed"]]

    print("=== Stellarator Field-Line MVP ===")
    print(
        f"Nfp={cfg.n_field_periods}, mode=(m={cfg.m_pol}, n={cfg.n_tor}), "
        f"epsilon={cfg.helical_perturbation:.4f}, lines={cfg.n_lines}, turns={cfg.n_toroidal_turns}"
    )
    print(f"confined_fraction = {out['confined_fraction']:.3f}")
    print(f"avg_iota_eff      = {out['avg_iota']:.4f}")
    print(f"avg_radial_span   = {out['avg_span']:.4f}")
    print(f"torch_loss        = {out['torch_loss']:.6e}")
    print(f"d(loss)/d(eps)    = {out['torch_grad']:.6e}")
    print(f"suggested_epsilon = {out['eps_suggested']:.6f}")

    print("\nPer-line summary:")
    print(preview.to_string(index=False, float_format=lambda z: f"{z:.6f}"))

    finite_ok = np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all()
    confinement_ok = out["confined_fraction"] >= 0.75
    iota_ok = 0.15 <= out["avg_iota"] <= 1.5
    fit_ok = bool((summary["iota_fit_r2"] > 0.95).all())
    grad_ok = abs(out["torch_grad"]) > 1e-8

    checks = [
        (bool(finite_ok), "finite_values"),
        (bool(confinement_ok), "confinement_ratio"),
        (bool(iota_ok), "iota_range"),
        (fit_ok, "linear_fit_quality"),
        (bool(grad_ok), "autograd_sensitivity"),
    ]

    passed = True
    for ok, name in checks:
        state = "PASS" if ok else "FAIL"
        print(f"{name}: {state}")
        if not ok:
            passed = False

    if not passed:
        raise SystemExit("Validation: FAIL")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
