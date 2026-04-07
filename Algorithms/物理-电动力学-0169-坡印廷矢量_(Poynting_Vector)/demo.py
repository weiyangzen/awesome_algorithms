"""Minimal runnable MVP for Poynting vector in vacuum electrodynamics.

The script builds a 1D traveling electromagnetic Gaussian pulse and verifies:
1) S = (1/mu0) E x B, with S_x = c * u for a vacuum plane wave profile.
2) Local energy conservation: du/dt + dS_x/dx ~= 0.
3) Global energy balance in a finite domain.
4) Energy transported through a probe plane vs analytical Gaussian integral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS0 * MU0)


@dataclass(frozen=True)
class PulseConfig:
    """Configuration for the 1D vacuum pulse experiment."""

    e0: float = 300.0
    sigma_m: float = 0.04
    x0_m: float = 0.25
    domain_length_m: float = 1.0
    nx: int = 1201
    t_end_s: float = 3.2e-9
    nt: int = 1601
    area_m2: float = 0.02
    probe_x_m: float = 0.70


def _assert_uniform_grid(grid: np.ndarray, name: str) -> float:
    """Return uniform step size and raise if the grid is not uniform."""
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError(f"{name} must be a 1D array with length >= 3")

    step = np.diff(grid)
    step0 = float(step[0])
    if step0 <= 0.0:
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(step, step0, rtol=1e-10, atol=1e-14):
        raise ValueError(f"{name} must be uniform in this MVP")
    return step0


def centered_difference(values: np.ndarray, step: float, axis: int) -> np.ndarray:
    """Finite difference on a uniform grid: central inside, one-sided at boundaries."""
    if step <= 0.0:
        raise ValueError("step must be positive")

    out = np.empty_like(values, dtype=np.float64)
    base = [slice(None)] * values.ndim

    interior = base.copy()
    interior[axis] = slice(1, -1)
    plus = base.copy()
    plus[axis] = slice(2, None)
    minus = base.copy()
    minus[axis] = slice(None, -2)
    out[tuple(interior)] = (values[tuple(plus)] - values[tuple(minus)]) / (2.0 * step)

    left0 = base.copy()
    left0[axis] = 0
    left1 = base.copy()
    left1[axis] = 1
    out[tuple(left0)] = (values[tuple(left1)] - values[tuple(left0)]) / step

    right0 = base.copy()
    right0[axis] = -1
    right1 = base.copy()
    right1[axis] = -2
    out[tuple(right0)] = (values[tuple(right0)] - values[tuple(right1)]) / step

    return out


def build_traveling_pulse(config: PulseConfig) -> Dict[str, np.ndarray]:
    """Build E, B, S, u fields for a right-going 1D vacuum pulse."""
    x = np.linspace(0.0, config.domain_length_m, config.nx, dtype=np.float64)
    t = np.linspace(0.0, config.t_end_s, config.nt, dtype=np.float64)

    _assert_uniform_grid(x, "x")
    _assert_uniform_grid(t, "t")

    xi = x[None, :] - C0 * t[:, None] - config.x0_m

    # Any smooth f(x-ct) is a 1D wave solution in vacuum; here choose a Gaussian envelope.
    e_y = config.e0 * np.exp(-(xi * xi) / (2.0 * config.sigma_m * config.sigma_m))
    b_z = e_y / C0

    e_vec = np.zeros((config.nt, config.nx, 3), dtype=np.float64)
    b_vec = np.zeros((config.nt, config.nx, 3), dtype=np.float64)
    e_vec[..., 1] = e_y
    b_vec[..., 2] = b_z

    s_vec = np.cross(e_vec, b_vec) / MU0
    u = 0.5 * (EPS0 * np.sum(e_vec * e_vec, axis=-1) + np.sum(b_vec * b_vec, axis=-1) / MU0)

    return {
        "x": x,
        "t": t,
        "E": e_vec,
        "B": b_vec,
        "S": s_vec,
        "u": u,
    }


def verify_pointwise_relations(fields: Dict[str, np.ndarray], config: PulseConfig) -> Dict[str, float]:
    """Check S_x = c u and positive propagation direction."""
    s_x = fields["S"][..., 0]
    u = fields["u"]
    target = C0 * u

    rel_err = float(np.max(np.abs(s_x - target)) / max(1e-15, np.max(np.abs(target))))

    pulse_mask = fields["E"][..., 1] > (0.20 * config.e0)
    direction_indicator = float(np.mean(s_x[pulse_mask]))

    assert rel_err < 1e-12, f"Pointwise S_x = c*u mismatch too large: {rel_err:.3e}"
    assert direction_indicator > 0.0, "Expected +x directed energy flow for this pulse"

    return {
        "pointwise_rel_err_s_eq_cu": rel_err,
        "direction_indicator_w_m2": direction_indicator,
    }


def verify_local_energy_continuity(fields: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Check du/dt + dS_x/dx ~= 0 on interior points."""
    x = fields["x"]
    t = fields["t"]
    u = fields["u"]
    s_x = fields["S"][..., 0]

    dx = _assert_uniform_grid(x, "x")
    dt = _assert_uniform_grid(t, "t")

    du_dt = centered_difference(u, dt, axis=0)
    dS_dx = centered_difference(s_x, dx, axis=1)

    residual = du_dt[:, 3:-3] + dS_dx[:, 3:-3]
    du_ref = du_dt[:, 3:-3]

    l2_rel = float(np.linalg.norm(residual) / max(1e-15, np.linalg.norm(du_ref)))
    max_abs = float(np.max(np.abs(residual)))

    assert l2_rel < 1.0e-3, f"Local continuity relative residual too large: {l2_rel:.3e}"

    return {
        "local_continuity_l2_rel": l2_rel,
        "local_continuity_max_abs": max_abs,
    }


def verify_global_energy_balance(fields: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Check d/dt int u dx + (S_right - S_left) ~= 0."""
    x = fields["x"]
    t = fields["t"]
    u = fields["u"]
    s_x = fields["S"][..., 0]

    dt = _assert_uniform_grid(t, "t")

    total_energy_per_area = np.trapezoid(u, x, axis=1)
    dU_dt = centered_difference(total_energy_per_area, dt, axis=0)
    net_out_flux = s_x[:, -1] - s_x[:, 0]

    residual = dU_dt + net_out_flux
    core = residual[3:-3]
    ref = dU_dt[3:-3]

    l2_rel = float(np.linalg.norm(core) / max(1e-15, np.linalg.norm(ref)))
    max_abs = float(np.max(np.abs(core)))

    assert l2_rel < 1.0e-3, f"Global energy balance relative residual too large: {l2_rel:.3e}"

    return {
        "global_balance_l2_rel": l2_rel,
        "global_balance_max_abs": max_abs,
    }


def gaussian_flux_integral_over_window(config: PulseConfig, t0: float, t1: float) -> float:
    """Analytical energy through x=probe for finite [t0, t1], including area."""
    y0 = (config.probe_x_m - C0 * t0 - config.x0_m) / config.sigma_m
    y1 = (config.probe_x_m - C0 * t1 - config.x0_m) / config.sigma_m

    coeff = config.area_m2 * EPS0 * (config.e0**2) * config.sigma_m * 0.5 * math.sqrt(math.pi)
    return coeff * (math.erf(y0) - math.erf(y1))


def verify_probe_energy_transport(fields: Dict[str, np.ndarray], config: PulseConfig) -> Dict[str, float]:
    """Compare numeric flux integral against closed-form Gaussian result."""
    x = fields["x"]
    t = fields["t"]
    s_x = fields["S"][..., 0]

    probe_index = int(np.argmin(np.abs(x - config.probe_x_m)))
    probe_x_actual = float(x[probe_index])

    numeric_energy = float(config.area_m2 * np.trapezoid(s_x[:, probe_index], t))
    analytic_energy = float(gaussian_flux_integral_over_window(config, float(t[0]), float(t[-1])))

    rel_err = abs(numeric_energy - analytic_energy) / max(1e-18, abs(analytic_energy))
    assert rel_err < 1.0e-3, f"Probe transported-energy mismatch too large: {rel_err:.3e}"

    return {
        "probe_x_m": probe_x_actual,
        "transport_energy_numeric_j": numeric_energy,
        "transport_energy_analytic_j": analytic_energy,
        "transport_energy_rel_err": rel_err,
    }


def run_demo(config: PulseConfig) -> pd.DataFrame:
    """Run all checks and return a compact metrics table."""
    fields = build_traveling_pulse(config)

    reports = []
    for section, result in (
        ("pointwise", verify_pointwise_relations(fields, config)),
        ("local_continuity", verify_local_energy_continuity(fields)),
        ("global_balance", verify_global_energy_balance(fields)),
        ("probe_transport", verify_probe_energy_transport(fields, config)),
    ):
        for key, value in result.items():
            reports.append({"section": section, "metric": key, "value": float(value)})

    return pd.DataFrame(reports)


def main() -> None:
    config = PulseConfig()
    metrics = run_demo(config)

    print("=== Poynting Vector MVP: Vacuum Gaussian Pulse ===")
    print("grid:")
    print(
        f"  nx={config.nx}, nt={config.nt}, L={config.domain_length_m:.3f} m, "
        f"t_end={config.t_end_s:.3e} s"
    )
    print(
        f"  e0={config.e0:.3f} V/m, sigma={config.sigma_m:.3f} m, x0={config.x0_m:.3f} m, "
        f"probe_x={config.probe_x_m:.3f} m"
    )

    print("\nmetrics:")
    for row in metrics.itertuples(index=False):
        print(f"  [{row.section:16s}] {row.metric:32s}: {row.value:.6e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
