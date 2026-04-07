"""Minimal runnable MVP for Landau theory of phase transitions.

This script demonstrates two standard Landau expansions:
1) Continuous (second-order) transition: b > 0, c = 0.
2) Discontinuous (first-order) transition: b < 0, c > 0.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LandauParams:
    """Coefficient set for homogeneous Landau free energy."""

    a: float
    b: float
    c: float
    tc: float
    h: float = 0.0
    m_max: float = 2.5


@dataclass(frozen=True)
class EquilibriumRow:
    """One temperature sample of the selected equilibrium state."""

    temperature: float
    m_eq: float
    abs_m_eq: float
    free_energy: float
    curvature: float
    susceptibility: float
    phase: str
    local_minima: int


def validate_inputs(
    temperatures: np.ndarray,
    params: LandauParams,
    grid_size: int,
    newton_tol: float,
    newton_max_iter: int,
) -> None:
    """Validate numerical configuration and physical coefficient constraints."""
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a non-empty 1D array")
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("temperatures contains non-finite values")
    if params.a <= 0:
        raise ValueError("Landau parameter a must be > 0")
    if params.c < 0:
        raise ValueError("Landau parameter c must be >= 0")
    if params.b < 0 and params.c <= 0:
        raise ValueError("for b < 0, c must be > 0 to keep free energy bounded")
    if params.m_max <= 0:
        raise ValueError("m_max must be > 0")
    if grid_size < 101:
        raise ValueError("grid_size must be >= 101")
    if grid_size % 2 == 0:
        raise ValueError("grid_size should be odd so m=0 is on the grid")
    if newton_tol <= 0:
        raise ValueError("newton_tol must be > 0")
    if newton_max_iter <= 0:
        raise ValueError("newton_max_iter must be > 0")


def free_energy(m: np.ndarray | float, temperature: float, params: LandauParams) -> np.ndarray | float:
    """Landau free energy density f(m, T) = a(T-Tc)m^2 + b m^4 + c m^6 - h m."""
    tau = temperature - params.tc
    return (
        params.a * tau * np.asarray(m) ** 2
        + params.b * np.asarray(m) ** 4
        + params.c * np.asarray(m) ** 6
        - params.h * np.asarray(m)
    )


def dfdm(m: float, temperature: float, params: LandauParams) -> float:
    """First derivative of Landau free energy with respect to order parameter m."""
    tau = temperature - params.tc
    return (
        2.0 * params.a * tau * m
        + 4.0 * params.b * m**3
        + 6.0 * params.c * m**5
        - params.h
    )


def d2fdm2(m: float, temperature: float, params: LandauParams) -> float:
    """Second derivative of Landau free energy with respect to m."""
    tau = temperature - params.tc
    return 2.0 * params.a * tau + 12.0 * params.b * m**2 + 30.0 * params.c * m**4


def newton_refine(
    m0: float,
    temperature: float,
    params: LandauParams,
    tol: float,
    max_iter: int,
) -> tuple[float, int, bool]:
    """Refine a stationary point using explicit Newton iteration on dF/dm = 0."""
    m = float(np.clip(m0, -params.m_max, params.m_max))

    for step in range(1, max_iter + 1):
        g = dfdm(m, temperature, params)
        hess = d2fdm2(m, temperature, params)

        if not math.isfinite(g) or not math.isfinite(hess):
            return m, step, False
        if abs(hess) < 1e-14:
            return m, step, False

        m_next = float(np.clip(m - g / hess, -params.m_max, params.m_max))
        if not math.isfinite(m_next):
            return m, step, False

        if abs(m_next - m) < tol and abs(g) < math.sqrt(tol):
            return m_next, step, True
        m = m_next

    return m, max_iter, False


def local_minima_seed_indices(fvals: np.ndarray) -> list[int]:
    """Return indices that look like local minima on a 1D grid."""
    n = fvals.size
    seeds: list[int] = []

    if n < 3:
        return [int(np.argmin(fvals))]

    if fvals[0] <= fvals[1]:
        seeds.append(0)
    for i in range(1, n - 1):
        if fvals[i] <= fvals[i - 1] and fvals[i] <= fvals[i + 1]:
            seeds.append(i)
    if fvals[-1] <= fvals[-2]:
        seeds.append(n - 1)

    if not seeds:
        seeds.append(int(np.argmin(fvals)))

    return seeds


def deduplicate_candidates(candidates: list[tuple[float, float, float]], tol: float = 1e-7) -> list[tuple[float, float, float]]:
    """Deduplicate minima candidates by close m value."""
    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda x: x[0])
    unique: list[tuple[float, float, float]] = []
    for item in candidates_sorted:
        if not unique or abs(item[0] - unique[-1][0]) > tol:
            unique.append(item)
        else:
            # Keep lower-free-energy one when two candidates map to same basin.
            if item[1] < unique[-1][1]:
                unique[-1] = item
    return unique


def equilibrium_at_temperature(
    temperature: float,
    params: LandauParams,
    grid_size: int,
    newton_tol: float,
    newton_max_iter: int,
) -> EquilibriumRow:
    """Find equilibrium order parameter by grid seeding + Newton refinement."""
    m_grid = np.linspace(-params.m_max, params.m_max, grid_size)
    f_grid = np.asarray(free_energy(m_grid, temperature, params), dtype=float)

    seed_ids = local_minima_seed_indices(f_grid)
    candidates: list[tuple[float, float, float]] = []

    for idx in seed_ids:
        m0 = float(m_grid[idx])
        m_refined, _, converged = newton_refine(
            m0=m0,
            temperature=temperature,
            params=params,
            tol=newton_tol,
            max_iter=newton_max_iter,
        )

        if not converged:
            m_refined = m0

        curvature = d2fdm2(m_refined, temperature, params)
        if curvature <= 0:
            continue

        f_val = float(free_energy(m_refined, temperature, params))
        candidates.append((m_refined, f_val, curvature))

    if not candidates:
        # Fallback: use raw grid minimum.
        idx = int(np.argmin(f_grid))
        m_best = float(m_grid[idx])
        f_best = float(f_grid[idx])
        curvature = max(d2fdm2(m_best, temperature, params), 1e-12)
        candidates = [(m_best, f_best, curvature)]

    candidates = deduplicate_candidates(candidates)
    # Tie-break chooses larger m if free energies are numerically equal.
    m_best, f_best, curvature = min(candidates, key=lambda x: (x[1], -x[0]))

    abs_m = abs(m_best)
    phase = "ordered" if abs_m > 1e-3 else "disordered"
    susceptibility = math.inf if curvature <= 1e-12 else 1.0 / curvature

    return EquilibriumRow(
        temperature=float(temperature),
        m_eq=float(m_best),
        abs_m_eq=float(abs_m),
        free_energy=float(f_best),
        curvature=float(curvature),
        susceptibility=float(susceptibility),
        phase=phase,
        local_minima=len(candidates),
    )


def scan_temperatures(
    temperatures: np.ndarray,
    params: LandauParams,
    grid_size: int,
    newton_tol: float,
    newton_max_iter: int,
) -> list[EquilibriumRow]:
    """Compute equilibrium state for each temperature in the scan list."""
    rows: list[EquilibriumRow] = []
    for temperature in temperatures:
        row = equilibrium_at_temperature(
            temperature=float(temperature),
            params=params,
            grid_size=grid_size,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
        )
        rows.append(row)
    return rows


def print_table(title: str, rows: list[EquilibriumRow]) -> None:
    """Print one scenario table in plain text."""
    print(title)
    header = (
        f"{'T':>6} {'m_eq':>10} {'|m_eq|':>10} {'phase':>11} "
        f"{'f_eq':>13} {'F_pp':>12} {'chi':>12} {'n_min':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        chi_str = "inf" if math.isinf(r.susceptibility) else f"{r.susceptibility: .6f}"
        print(
            f"{r.temperature:6.3f} {r.m_eq:10.6f} {r.abs_m_eq:10.6f} {r.phase:>11} "
            f"{r.free_energy:13.6f} {r.curvature:12.6f} {chi_str:>12} {r.local_minima:7d}"
        )


def analytic_abs_m_second_order(temperature: float, params: LandauParams) -> float:
    """Closed-form |m| for c=0, b>0, h=0 continuous Landau model."""
    if not (params.b > 0 and abs(params.c) < 1e-15 and abs(params.h) < 1e-15):
        raise ValueError("analytic formula requires b>0, c=0, h=0")
    if temperature >= params.tc:
        return 0.0
    value = params.a * (params.tc - temperature) / (2.0 * params.b)
    return math.sqrt(max(value, 0.0))


def run_sanity_checks_continuous(rows: list[EquilibriumRow], params: LandauParams) -> None:
    """Basic checks for the second-order scenario."""
    below = [r for r in rows if r.temperature < params.tc]
    above = [r for r in rows if r.temperature > params.tc]
    if not below or not above:
        raise RuntimeError("continuous scan must include both below and above Tc")

    near_below = max(below, key=lambda r: r.temperature)
    near_above = min(above, key=lambda r: r.temperature)

    if near_below.abs_m_eq < 0.05:
        raise AssertionError("expected non-zero order parameter just below Tc")
    if near_above.abs_m_eq > 0.08:
        raise AssertionError("expected near-zero order parameter just above Tc")


def run_sanity_checks_first_order(rows: list[EquilibriumRow], params: LandauParams) -> None:
    """Basic checks for the first-order scenario (jump detection)."""
    diffs = []
    for i in range(1, len(rows)):
        dm = abs(rows[i].abs_m_eq - rows[i - 1].abs_m_eq)
        diffs.append((dm, rows[i - 1].temperature, rows[i].temperature))

    if not diffs:
        raise RuntimeError("not enough points for jump detection")

    max_jump, t_left, t_right = max(diffs, key=lambda x: x[0])
    if max_jump < 0.10:
        raise AssertionError("expected a visible discontinuous jump in first-order case")

    t_coexist = params.tc + (params.b * params.b) / (4.0 * params.a * params.c)
    if not (min(r.temperature for r in rows) <= t_coexist <= max(r.temperature for r in rows)):
        raise AssertionError("temperature grid should cover the predicted coexistence temperature")

    print(
        "First-order jump diagnostic: "
        f"largest |Δm|={max_jump:.4f} between T={t_left:.3f} and T={t_right:.3f}; "
        f"predicted coexistence T*={t_coexist:.3f}"
    )


def main() -> None:
    """Run deterministic Landau-theory demos for two coefficient regimes."""
    np.set_printoptions(precision=6, suppress=True)

    grid_size = 3001
    newton_tol = 1e-12
    newton_max_iter = 80

    # Scenario A: continuous transition, b > 0, c = 0.
    params_cont = LandauParams(a=1.0, b=1.0, c=0.0, tc=1.0, h=0.0, m_max=2.0)
    temps_cont = np.array([0.60, 0.70, 0.80, 0.90, 0.98, 1.00, 1.02, 1.10, 1.20, 1.40])
    validate_inputs(temps_cont, params_cont, grid_size, newton_tol, newton_max_iter)

    rows_cont = scan_temperatures(temps_cont, params_cont, grid_size, newton_tol, newton_max_iter)
    print("Landau Theory MVP")
    print(
        "Scenario A (continuous): "
        f"a={params_cont.a}, b={params_cont.b}, c={params_cont.c}, Tc={params_cont.tc}, h={params_cont.h}"
    )
    print_table("Equilibrium scan", rows_cont)

    # Compare near Tc with analytic |m| = sqrt(a(Tc-T)/(2b)).
    print("\nNear-critical comparison with analytic second-order formula:")
    for row in sorted((r for r in rows_cont if r.temperature < params_cont.tc), key=lambda x: x.temperature)[-3:]:
        m_analytic = analytic_abs_m_second_order(row.temperature, params_cont)
        rel_err = abs(row.abs_m_eq - m_analytic) / max(m_analytic, 1e-12)
        print(
            f"T={row.temperature:.3f}, |m_num|={row.abs_m_eq:.6f}, "
            f"|m_analytic|={m_analytic:.6f}, rel_err={rel_err:.2%}"
        )

    run_sanity_checks_continuous(rows_cont, params_cont)

    # Scenario B: first-order transition, b < 0, c > 0.
    params_first = LandauParams(a=1.0, b=-1.0, c=1.0, tc=1.0, h=0.0, m_max=2.5)
    temps_first = np.array([0.80, 0.90, 1.00, 1.10, 1.20, 1.24, 1.26, 1.30, 1.40, 1.50])
    validate_inputs(temps_first, params_first, grid_size, newton_tol, newton_max_iter)

    rows_first = scan_temperatures(temps_first, params_first, grid_size, newton_tol, newton_max_iter)
    print()
    print(
        "Scenario B (first-order): "
        f"a={params_first.a}, b={params_first.b}, c={params_first.c}, Tc={params_first.tc}, h={params_first.h}"
    )
    print_table("Equilibrium scan", rows_first)

    run_sanity_checks_first_order(rows_first, params_first)
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
