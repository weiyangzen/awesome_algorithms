"""Minimal runnable MVP for BCS theory (s-wave, weak-coupling).

The script solves the finite-temperature BCS gap equation in a transparent way:
1) Compute the pairing integral on an explicit energy grid.
2) Solve the linearized gap equation to obtain Tc.
3) Solve Delta(T) by bisection for each temperature.
4) Report consistency checks such as 2Delta0/Tc ~ 3.53.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BCSConfig:
    """Configuration for an isotropic weak-coupling BCS MVP."""

    coupling_lambda: float = 0.30  # dimensionless: lambda = V * N(0)
    debye_cutoff: float = 1.0  # omega_D, energy unit
    n_energy_grid: int = 6000
    energy_min_positive: float = 1e-8
    t_floor: float = 1e-8
    tc_search_low: float = 1e-5
    tc_search_high: float = 0.6
    delta_upper_init: float = 0.2
    n_temperatures: int = 32
    temperature_min_factor: float = 0.05  # Tmin = factor * Tc
    temperature_max_factor: float = 1.20  # Tmax = factor * Tc
    root_tol: float = 1e-10
    max_bisect_iter: int = 120


def build_energy_grid(cfg: BCSConfig) -> np.ndarray:
    """Build a nonuniform energy grid dense near xi=0 for logarithmic kernels."""
    tail = np.geomspace(cfg.energy_min_positive, cfg.debye_cutoff, cfg.n_energy_grid - 1)
    return np.concatenate(([0.0], tail))


def pairing_integral(delta: float, temperature: float, xi_grid: np.ndarray, cfg: BCSConfig) -> float:
    """Return I(delta, T) = int_0^wD tanh(E/2T)/E dxi, E=sqrt(xi^2+delta^2)."""
    delta = float(max(delta, 0.0))
    temperature = float(max(temperature, 0.0))

    if delta <= 1e-15:
        if temperature <= cfg.t_floor:
            raise ValueError("Linearized integral requires positive temperature.")
        kernel = np.empty_like(xi_grid)
        kernel[0] = 1.0 / (2.0 * temperature)
        xi_tail = xi_grid[1:]
        kernel[1:] = np.tanh(xi_tail / (2.0 * temperature)) / xi_tail
        return float(np.trapezoid(kernel, xi_grid))

    energy = np.sqrt(xi_grid * xi_grid + delta * delta)
    if temperature <= cfg.t_floor:
        kernel = 1.0 / energy
    else:
        kernel = np.tanh(energy / (2.0 * temperature)) / energy
    return float(np.trapezoid(kernel, xi_grid))


def gap_residual(delta: float, temperature: float, xi_grid: np.ndarray, cfg: BCSConfig) -> float:
    """BCS self-consistency residual: f = 1/lambda - I(delta, T)."""
    return 1.0 / cfg.coupling_lambda - pairing_integral(delta, temperature, xi_grid, cfg)


def bisect_root(
    func,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> float:
    """Generic bisection with sign-bracketing guarantee."""
    f_left = float(func(left))
    f_right = float(func(right))

    if abs(f_left) <= tol:
        return float(left)
    if abs(f_right) <= tol:
        return float(right)
    if f_left * f_right > 0.0:
        raise ValueError("Bisection interval does not bracket a root.")

    l = float(left)
    r = float(right)
    for _ in range(max_iter):
        mid = 0.5 * (l + r)
        f_mid = float(func(mid))
        if abs(f_mid) <= tol or abs(r - l) <= tol:
            return mid

        if f_left * f_mid <= 0.0:
            r = mid
            f_right = f_mid
        else:
            l = mid
            f_left = f_mid

    return 0.5 * (l + r)


def solve_critical_temperature(cfg: BCSConfig, xi_grid: np.ndarray) -> float:
    """Solve Tc from the linearized BCS equation (delta->0)."""

    def linearized_residual(temp: float) -> float:
        return gap_residual(delta=0.0, temperature=temp, xi_grid=xi_grid, cfg=cfg)

    low = max(cfg.tc_search_low, cfg.t_floor * 10.0)
    high = cfg.tc_search_high

    f_low = linearized_residual(low)
    if f_low > 0.0:
        while f_low > 0.0 and low > cfg.t_floor * 10.0:
            low *= 0.5
            f_low = linearized_residual(low)

    f_high = linearized_residual(high)
    while f_high < 0.0:
        high *= 1.6
        if high > 5.0 * cfg.debye_cutoff:
            raise RuntimeError("Failed to bracket Tc in the linearized equation.")
        f_high = linearized_residual(high)

    return bisect_root(
        func=linearized_residual,
        left=low,
        right=high,
        tol=cfg.root_tol,
        max_iter=cfg.max_bisect_iter,
    )


def solve_gap_at_temperature(temperature: float, cfg: BCSConfig, xi_grid: np.ndarray) -> float:
    """Solve Delta(T) >= 0 from full BCS equation."""

    def residual(delta: float) -> float:
        return gap_residual(delta=delta, temperature=temperature, xi_grid=xi_grid, cfg=cfg)

    r_zero = residual(0.0)
    if r_zero >= 0.0:
        return 0.0

    high = cfg.delta_upper_init
    r_high = residual(high)
    while r_high < 0.0:
        high *= 1.8
        if high > 20.0 * cfg.debye_cutoff:
            raise RuntimeError("Failed to bracket nonzero gap root.")
        r_high = residual(high)

    return bisect_root(
        func=residual,
        left=0.0,
        right=high,
        tol=cfg.root_tol,
        max_iter=cfg.max_bisect_iter,
    )


def run_temperature_scan(cfg: BCSConfig, xi_grid: np.ndarray, tc_numeric: float) -> pd.DataFrame:
    """Compute Delta(T) and derived quantities over a temperature grid."""
    t_min = cfg.temperature_min_factor * tc_numeric
    t_max = cfg.temperature_max_factor * tc_numeric
    temperatures = np.linspace(t_min, t_max, cfg.n_temperatures)

    rows: list[dict[str, float]] = []
    for t in temperatures:
        temperature = float(t)
        delta = solve_gap_at_temperature(temperature, cfg, xi_grid)
        residual_value = gap_residual(delta, temperature, xi_grid, cfg)

        rows.append(
            {
                "temperature": temperature,
                "reduced_T": temperature / tc_numeric,
                "gap_delta": delta,
                "gap_over_cutoff": delta / cfg.debye_cutoff,
                "quasiparticle_min_energy": delta,
                "condensation_energy_density": -0.5 * delta * delta,
                "residual_at_solution": residual_value,
            }
        )

    return pd.DataFrame(rows).sort_values("temperature", ignore_index=True)


def run_consistency_checks(
    df: pd.DataFrame,
    tc_numeric: float,
    tc_weak_coupling: float,
    delta0_weak_coupling: float,
) -> dict[str, float]:
    """Validate major BCS trends and return summary metrics."""
    delta0_numeric = float(df.iloc[0]["gap_delta"])
    ratio = 2.0 * delta0_numeric / tc_numeric
    tc_rel_error = abs(tc_numeric - tc_weak_coupling) / tc_weak_coupling

    below_tc = df[df["temperature"] < 0.98 * tc_numeric]
    if len(below_tc) >= 3:
        gaps = below_tc["gap_delta"].to_numpy(dtype=float)
        assert np.all(np.diff(gaps) <= 2e-3), "Delta(T) should decrease with T below Tc."

    high_t_gap = float(df.iloc[-1]["gap_delta"])

    assert delta0_numeric > 0.25 * delta0_weak_coupling
    assert high_t_gap < 1e-5
    assert 2.8 < ratio < 4.4
    assert tc_rel_error < 0.25

    return {
        "delta0_numeric": delta0_numeric,
        "ratio_2delta0_over_tc": ratio,
        "tc_relative_error": tc_rel_error,
        "high_temperature_gap": high_t_gap,
    }


def main() -> None:
    cfg = BCSConfig()
    xi_grid = build_energy_grid(cfg)

    tc_numeric = solve_critical_temperature(cfg, xi_grid)
    tc_weak_coupling = 1.13 * cfg.debye_cutoff * np.exp(-1.0 / cfg.coupling_lambda)
    delta0_weak_coupling = 2.0 * cfg.debye_cutoff * np.exp(-1.0 / cfg.coupling_lambda)

    df = run_temperature_scan(cfg, xi_grid, tc_numeric)
    checks = run_consistency_checks(df, tc_numeric, tc_weak_coupling, delta0_weak_coupling)

    print("=== BCS Theory MVP (s-wave, weak-coupling) ===")
    print(
        "config:",
        f"lambda={cfg.coupling_lambda}, omega_D={cfg.debye_cutoff}, "
        f"energy_grid={cfg.n_energy_grid}",
    )
    print(
        f"Tc (numeric)={tc_numeric:.8f}, "
        f"Tc (weak-coupling approx)={tc_weak_coupling:.8f}, "
        f"Delta0 approx={delta0_weak_coupling:.8f}"
    )
    print()

    with pd.option_context("display.width", 180, "display.precision", 8):
        print(df.to_string(index=False))
    print()

    print("Derived checks:")
    for key, value in checks.items():
        print(f"- {key}: {value:.8f}")


if __name__ == "__main__":
    main()
