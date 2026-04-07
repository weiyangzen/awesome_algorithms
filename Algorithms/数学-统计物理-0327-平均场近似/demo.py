"""Minimal runnable MVP for mean-field approximation in the Ising model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IterationLog:
    """One fixed-point solve result for a single (T, seed)."""

    temperature: float
    beta: float
    seed_m0: float
    m_star: float
    free_energy: float
    iterations: int
    residual: float
    converged: bool


@dataclass(frozen=True)
class EquilibriumRow:
    """Equilibrium branch selected by minimum free energy at a given T."""

    temperature: float
    beta: float
    m_eq: float
    abs_m_eq: float
    phase: str
    free_energy: float
    susceptibility: float
    iterations: int
    residual: float
    chosen_seed: float


def validate_config(
    temperatures: np.ndarray,
    J: float,
    z: float,
    tol: float,
    max_iter: int,
    damping: float,
) -> None:
    """Validate scalar and vector inputs for the mean-field scan."""
    if J <= 0:
        raise ValueError("J must be > 0 for ferromagnetic mean-field Ising in this MVP.")
    if z <= 0:
        raise ValueError("z must be > 0.")
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("temperatures must be a non-empty 1D array.")
    if not np.all(np.isfinite(temperatures)):
        raise ValueError("temperatures contains non-finite values.")
    if np.any(temperatures <= 0):
        raise ValueError("all temperatures must be > 0.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if not (0 < damping <= 1):
        raise ValueError("damping must be in (0, 1].")


def mean_field_map(m: float, beta: float, J: float, z: float, h: float) -> float:
    """Self-consistency map m -> tanh(beta * (J z m + h))."""
    return math.tanh(beta * (J * z * m + h))


def free_energy_per_spin(m: float, beta: float, J: float, z: float, h: float) -> float:
    """Mean-field free-energy density for Ising model."""
    x = beta * (J * z * m + h)
    # log(2 cosh(x)) = logaddexp(x, -x), numerically stable for larger |x|.
    log_2cosh = float(np.logaddexp(x, -x))
    return 0.5 * J * z * m * m - (1.0 / beta) * log_2cosh


def solve_fixed_point(
    beta: float,
    J: float,
    z: float,
    h: float,
    m0: float,
    tol: float,
    max_iter: int,
    damping: float,
) -> tuple[float, int, float, bool]:
    """Solve m = tanh(beta(J z m + h)) by damped fixed-point iteration."""
    m = float(np.clip(m0, -1.0, 1.0))
    residual = math.inf

    for step in range(1, max_iter + 1):
        mapped = mean_field_map(m, beta, J, z, h)
        m_next = (1.0 - damping) * m + damping * mapped
        m_next = float(np.clip(m_next, -1.0, 1.0))

        if not math.isfinite(m_next):
            raise RuntimeError("non-finite iterate encountered in fixed-point solver.")

        residual = abs(m_next - m)
        if residual < tol:
            return m_next, step, residual, True

        m = m_next

    return m, max_iter, residual, False


def scan_temperatures(
    temperatures: np.ndarray,
    J: float,
    z: float,
    h: float,
    seeds: Iterable[float],
    tol: float,
    max_iter: int,
    damping: float,
) -> list[IterationLog]:
    """Run fixed-point solves for every temperature and initial seed."""
    logs: list[IterationLog] = []
    for T in temperatures:
        beta = 1.0 / float(T)
        for seed in seeds:
            m_star, iters, residual, converged = solve_fixed_point(
                beta=beta,
                J=J,
                z=z,
                h=h,
                m0=float(seed),
                tol=tol,
                max_iter=max_iter,
                damping=damping,
            )
            fval = free_energy_per_spin(m_star, beta, J, z, h)
            logs.append(
                IterationLog(
                    temperature=float(T),
                    beta=beta,
                    seed_m0=float(seed),
                    m_star=m_star,
                    free_energy=fval,
                    iterations=iters,
                    residual=residual,
                    converged=converged,
                )
            )
    return logs


def mean_field_susceptibility(m: float, beta: float, J: float, z: float) -> float:
    """Susceptibility from linearized self-consistency equation."""
    one_minus_m2 = 1.0 - m * m
    numerator = beta * one_minus_m2
    denominator = 1.0 - beta * J * z * one_minus_m2
    if abs(denominator) < 1e-12:
        return math.inf
    return numerator / denominator


def build_equilibrium_rows(
    logs: list[IterationLog],
    J: float,
    z: float,
    magnetization_threshold: float = 1e-3,
) -> list[EquilibriumRow]:
    """Choose equilibrium branch (minimum free energy) per temperature."""
    grouped: dict[float, list[IterationLog]] = {}
    for item in logs:
        grouped.setdefault(item.temperature, []).append(item)

    rows: list[EquilibriumRow] = []
    for T in sorted(grouped.keys()):
        candidates = grouped[T]
        converged = [c for c in candidates if c.converged]
        pool = converged if converged else candidates
        best = min(pool, key=lambda c: (c.free_energy, c.residual, abs(c.m_star)))

        abs_m = abs(best.m_star)
        phase = "ordered" if abs_m > magnetization_threshold else "disordered"
        chi = mean_field_susceptibility(best.m_star, best.beta, J, z)

        rows.append(
            EquilibriumRow(
                temperature=T,
                beta=best.beta,
                m_eq=best.m_star,
                abs_m_eq=abs_m,
                phase=phase,
                free_energy=best.free_energy,
                susceptibility=chi,
                iterations=best.iterations,
                residual=best.residual,
                chosen_seed=best.seed_m0,
            )
        )

    return rows


def print_equilibrium_table(rows: list[EquilibriumRow]) -> None:
    """Print a compact table for the temperature scan."""
    header = (
        f"{'T':>6} {'beta':>8} {'m_eq':>10} {'|m_eq|':>10} {'phase':>11} "
        f"{'f_eq':>13} {'chi':>12} {'iter':>6} {'residual':>12} {'seed':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        chi_str = "inf" if math.isinf(r.susceptibility) else f"{r.susceptibility: .5f}"
        print(
            f"{r.temperature:6.3f} {r.beta:8.4f} {r.m_eq:10.6f} {r.abs_m_eq:10.6f} "
            f"{r.phase:>11} {r.free_energy:13.6f} {chi_str:>12} {r.iterations:6d} "
            f"{r.residual:12.3e} {r.chosen_seed:8.3f}"
        )


def run_sanity_checks(rows: list[EquilibriumRow], Tc: float) -> None:
    """Simple physics sanity checks around Tc."""
    below = [r for r in rows if r.temperature < Tc]
    above = [r for r in rows if r.temperature > Tc]
    if not below or not above:
        raise RuntimeError("temperature grid must include both below-Tc and above-Tc points.")

    near_below = max(below, key=lambda r: r.temperature)
    near_above = min(above, key=lambda r: r.temperature)

    if near_below.abs_m_eq < 0.05:
        raise AssertionError("expected non-zero spontaneous magnetization just below Tc.")
    if near_above.abs_m_eq > 0.10:
        raise AssertionError("expected near-zero magnetization just above Tc.")


def main() -> None:
    """Run a deterministic mean-field Ising temperature scan."""
    J = 1.0
    z = 4.0
    h = 0.0

    temperatures = np.array([2.0, 2.5, 3.0, 3.5, 3.8, 3.9, 4.0, 4.1, 4.5, 5.0], dtype=float)
    seeds = (-0.95, 0.0, 0.95)

    tol = 1e-12
    max_iter = 20_000
    damping = 0.6

    validate_config(temperatures, J=J, z=z, tol=tol, max_iter=max_iter, damping=damping)

    Tc = J * z
    print("Mean-field Ising MVP (k_B = 1)")
    print(f"J={J}, z={z}, h={h}, Tc={Tc:.3f}")
    print(f"temperatures={temperatures.tolist()}")
    print(f"seeds={list(seeds)}, tol={tol}, max_iter={max_iter}, damping={damping}")
    print()

    logs = scan_temperatures(
        temperatures=temperatures,
        J=J,
        z=z,
        h=h,
        seeds=seeds,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )

    rows = build_equilibrium_rows(logs=logs, J=J, z=z)
    print_equilibrium_table(rows)

    converged_count = sum(1 for x in logs if x.converged)
    total_count = len(logs)
    print()
    print(f"Candidate convergence: {converged_count}/{total_count} = {converged_count/total_count:.2%}")

    # Compare one point below Tc to Landau expansion m ~ sqrt(3*(Tc-T)/Tc).
    below_tc_rows = [r for r in rows if r.temperature < Tc]
    if below_tc_rows:
        near_tc = max(below_tc_rows, key=lambda r: r.temperature)
        reduced = max(0.0, (Tc - near_tc.temperature) / Tc)
        landau_m = math.sqrt(3.0 * reduced)
        rel_err = abs(near_tc.abs_m_eq - landau_m) / max(landau_m, 1e-12)
        print(
            "Near-critical check: "
            f"T={near_tc.temperature:.3f}, |m_num|={near_tc.abs_m_eq:.6f}, "
            f"|m_landau|={landau_m:.6f}, rel_err={rel_err:.2%}"
        )

    run_sanity_checks(rows, Tc=Tc)
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
