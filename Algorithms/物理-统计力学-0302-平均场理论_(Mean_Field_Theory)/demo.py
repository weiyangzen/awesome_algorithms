"""Minimal runnable MVP for Mean Field Theory (Ising model).

The script implements a fully transparent mean-field workflow:
1) Solve the self-consistency equation m = tanh(beta * (z J m + h)).
2) Enumerate all fixed points by scan + bisection (no black-box solver).
3) Classify stable points and choose equilibrium by minimal free energy.
4) Scan temperature and verify the expected phase-transition trend near Tc=zJ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MeanFieldConfig:
    coordination_number: int = 4
    coupling_j: float = 1.0
    external_field: float = 0.0
    temperature_min: float = 1.2
    temperature_max: float = 6.0
    n_temperatures: int = 33
    scan_points: int = 901
    root_tol: float = 1e-12
    max_bisect_iter: int = 120


def residual(m: float | np.ndarray, beta: float, z: int, j: float, h: float) -> float | np.ndarray:
    """Self-consistency residual g(m)=m-tanh(beta*(zjm+h))."""
    return m - np.tanh(beta * (z * j * m + h))


def bisect_root(
    left: float,
    right: float,
    beta: float,
    z: int,
    j: float,
    h: float,
    tol: float,
    max_iter: int,
) -> float:
    """Bisection root solve for residual on [left, right] with sign change."""
    f_left = float(residual(left, beta, z, j, h))
    f_right = float(residual(right, beta, z, j, h))

    if abs(f_left) <= tol:
        return left
    if abs(f_right) <= tol:
        return right
    if f_left * f_right > 0.0:
        raise ValueError("Bisection interval does not bracket a root.")

    l = left
    r = right
    for _ in range(max_iter):
        mid = 0.5 * (l + r)
        f_mid = float(residual(mid, beta, z, j, h))

        if abs(f_mid) <= tol or abs(r - l) <= tol:
            return mid

        if f_left * f_mid <= 0.0:
            r = mid
            f_right = f_mid
        else:
            l = mid
            f_left = f_mid

    return 0.5 * (l + r)


def deduplicate_sorted(values: list[float], tol: float = 1e-8) -> list[float]:
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        if abs(v - out[-1]) > tol:
            out.append(float(v))
    return out


def stability_slope(m: float, beta: float, z: int, j: float) -> float:
    """Derivative of fixed-point map tanh(beta*(zjm+h)) at m."""
    return beta * z * j * (1.0 - m * m)


def free_energy_density(m: float, temperature: float, z: int, j: float, h: float) -> float:
    """Mean-field free energy per spin for Ising model."""
    m_clip = float(np.clip(m, -1.0 + 1e-15, 1.0 - 1e-15))
    p_up = 0.5 * (1.0 + m_clip)
    p_dn = 0.5 * (1.0 - m_clip)
    entropy_part = p_up * math.log(p_up) + p_dn * math.log(p_dn)
    interaction_part = -0.5 * z * j * m_clip * m_clip - h * m_clip
    return interaction_part + temperature * entropy_part


def find_fixed_points(temperature: float, cfg: MeanFieldConfig) -> tuple[np.ndarray, np.ndarray]:
    """Find all fixed points by sign-scan + bisection and classify stability."""
    beta = 1.0 / temperature
    z = cfg.coordination_number
    j = cfg.coupling_j
    h = cfg.external_field

    m_grid = np.linspace(-0.999999, 0.999999, cfg.scan_points)
    g_grid = residual(m_grid, beta, z, j, h)

    candidates: list[float] = []

    near_zero_idx = np.where(np.abs(g_grid) < 1e-8)[0]
    for idx in near_zero_idx:
        candidates.append(float(m_grid[idx]))

    for i in range(cfg.scan_points - 1):
        g1 = float(g_grid[i])
        g2 = float(g_grid[i + 1])
        if g1 == 0.0 or g2 == 0.0:
            continue
        if g1 * g2 < 0.0:
            root = bisect_root(
                left=float(m_grid[i]),
                right=float(m_grid[i + 1]),
                beta=beta,
                z=z,
                j=j,
                h=h,
                tol=cfg.root_tol,
                max_iter=cfg.max_bisect_iter,
            )
            candidates.append(root)

    roots = np.array(deduplicate_sorted(sorted(candidates), tol=5e-7), dtype=float)
    slopes = np.array([stability_slope(m, beta, z, j) for m in roots], dtype=float)
    stable = np.abs(slopes) < 1.0 - 1e-12
    return roots, stable


def select_equilibrium_root(
    roots: np.ndarray,
    stable_mask: np.ndarray,
    temperature: float,
    cfg: MeanFieldConfig,
) -> float:
    """Pick physical equilibrium root by minimal free energy among stable roots.

    If h=0 and there is free-energy degeneracy (±m), choose +m by convention.
    """
    if len(roots) == 0:
        raise ValueError("No fixed points found; check scan range or parameters.")

    z = cfg.coordination_number
    j = cfg.coupling_j
    h = cfg.external_field

    if np.any(stable_mask):
        pool = roots[stable_mask]
    else:
        pool = roots

    energies = np.array([free_energy_density(m, temperature, z, j, h) for m in pool], dtype=float)
    e_min = float(np.min(energies))
    tie_idx = np.where(np.abs(energies - e_min) < 1e-10)[0]
    tie_roots = pool[tie_idx]

    if abs(h) < 1e-14:
        return float(np.max(tie_roots))
    return float(tie_roots[0])


def susceptibility_from_root(m: float, temperature: float, z: int, j: float) -> float:
    """Analytic dm/dh for mean-field Ising at a self-consistent root."""
    beta = 1.0 / temperature
    numerator = beta * (1.0 - m * m)
    denom = 1.0 - beta * z * j * (1.0 - m * m)
    if abs(denom) < 1e-12:
        return float("inf")
    return float(numerator / denom)


def run_temperature_scan(cfg: MeanFieldConfig) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    temperatures = np.linspace(cfg.temperature_min, cfg.temperature_max, cfg.n_temperatures)

    for t in temperatures:
        temperature = float(t)
        roots, stable = find_fixed_points(temperature, cfg)
        m_eq = select_equilibrium_root(roots, stable, temperature, cfg)

        stable_roots = roots[stable] if np.any(stable) else roots
        pos_branch = float(np.max(stable_roots))
        neg_branch = float(np.min(stable_roots))

        chi = susceptibility_from_root(
            m=m_eq,
            temperature=temperature,
            z=cfg.coordination_number,
            j=cfg.coupling_j,
        )

        rows.append(
            {
                "temperature": temperature,
                "m_equilibrium": m_eq,
                "m_abs": abs(m_eq),
                "m_positive_branch": pos_branch,
                "m_negative_branch": neg_branch,
                "susceptibility": chi,
                "n_fixed_points": float(len(roots)),
                "n_stable_points": float(np.sum(stable)),
            }
        )

    return pd.DataFrame(rows).sort_values("temperature", ignore_index=True)


def estimate_tc_from_susceptibility(df: pd.DataFrame) -> float:
    chi = df["susceptibility"].to_numpy(dtype=float)
    safe = np.where(np.isfinite(chi), chi, -np.inf)
    idx = int(np.argmax(safe))
    return float(df.loc[idx, "temperature"])


def main() -> None:
    cfg = MeanFieldConfig()
    df = run_temperature_scan(cfg)

    tc_theory = cfg.coordination_number * cfg.coupling_j
    tc_numeric = estimate_tc_from_susceptibility(df)

    print("=== Mean Field Theory MVP (Ising) ===")
    print(
        "config:",
        f"z={cfg.coordination_number}, J={cfg.coupling_j}, h={cfg.external_field},",
        f"T in [{cfg.temperature_min}, {cfg.temperature_max}] with {cfg.n_temperatures} points",
    )
    print()
    with pd.option_context("display.width", 200, "display.precision", 5):
        print(df.to_string(index=False))
    print()
    print(f"Theoretical critical temperature Tc = zJ = {tc_theory:.5f}")
    print(f"Numerical susceptibility-peak Tc*   = {tc_numeric:.5f}")

    low_temp_row = df.iloc[0]
    high_temp_row = df.iloc[-1]

    assert float(low_temp_row["m_abs"]) > 0.5
    assert float(high_temp_row["m_abs"]) < 0.2
    assert float(df[df["temperature"] < tc_theory]["n_stable_points"].max()) >= 2.0
    assert abs(tc_numeric - tc_theory) < 1.0

    print("Checks passed: low-T ordered phase, high-T disordered phase, and Tc vicinity are consistent.")


if __name__ == "__main__":
    main()
