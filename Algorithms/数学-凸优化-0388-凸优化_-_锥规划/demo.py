"""Conic optimization MVP: solve a second-order cone program (SOCP) by PGD.

Problem form used in this demo:
    minimize_z  0.5 * ||M z - d||_2^2 + g^T z
    subject to  z in Q^n

where Q^n is the second-order cone (SOC):
    Q^n = {(t, u) | ||u||_2 <= t}

The script is deterministic and requires no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


HistoryItem = Tuple[int, float, float, float, float]


@dataclass
class SOCPResult:
    z: np.ndarray
    history: List[HistoryItem]
    converged: bool
    iters_used: int
    step_size: float


def validate_problem(m: np.ndarray, d: np.ndarray, g: np.ndarray) -> None:
    """Validate shapes and numeric sanity for the SOCP instance."""
    if m.ndim != 2:
        raise ValueError(f"M must be 2D, got shape={m.shape}.")
    if d.ndim != 1:
        raise ValueError(f"d must be 1D, got shape={d.shape}.")
    if g.ndim != 1:
        raise ValueError(f"g must be 1D, got shape={g.shape}.")

    n_obs, n_dim = m.shape
    if n_obs == 0 or n_dim < 2:
        raise ValueError("M must have non-zero rows and at least 2 columns (t + u).")
    if d.shape[0] != n_obs:
        raise ValueError(f"Dimension mismatch: d has len={d.shape[0]} but M has {n_obs} rows.")
    if g.shape[0] != n_dim:
        raise ValueError(f"Dimension mismatch: g has len={g.shape[0]} but M has {n_dim} cols.")

    if not np.all(np.isfinite(m)):
        raise ValueError("M contains non-finite values.")
    if not np.all(np.isfinite(d)):
        raise ValueError("d contains non-finite values.")
    if not np.all(np.isfinite(g)):
        raise ValueError("g contains non-finite values.")


def soc_feasibility_gap(z: np.ndarray) -> float:
    """Return max(||u|| - t, 0), equals 0 iff z in SOC (up to tolerance)."""
    t = float(z[0])
    u_norm = float(np.linalg.norm(z[1:]))
    return max(u_norm - t, 0.0)


def project_soc(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the second-order cone Q^n.

    For v = (t, u), projection has closed-form:
    1) if ||u|| <= t      -> v
    2) if ||u|| <= -t     -> 0
    3) otherwise          -> (a, a/||u|| * u), a = (||u|| + t)/2
    """
    t = float(v[0])
    u = v[1:]
    u_norm = float(np.linalg.norm(u))

    if u_norm <= t:
        return v.copy()
    if u_norm <= -t:
        return np.zeros_like(v)

    a = 0.5 * (u_norm + t)
    out = np.empty_like(v)
    out[0] = a
    out[1:] = (a / u_norm) * u
    return out


def objective(m: np.ndarray, d: np.ndarray, g: np.ndarray, z: np.ndarray) -> float:
    residual = m @ z - d
    return 0.5 * float(np.dot(residual, residual)) + float(np.dot(g, z))


def gradient(m: np.ndarray, d: np.ndarray, g: np.ndarray, z: np.ndarray) -> np.ndarray:
    return m.T @ (m @ z - d) + g


def lipschitz_constant(m: np.ndarray) -> float:
    """L for grad f(z) = M^T(Mz-d)+g is ||M||_2^2."""
    spectral_norm = float(np.linalg.norm(m, ord=2))
    return max(spectral_norm * spectral_norm, 1e-12)


def projected_gradient_mapping(
    z: np.ndarray,
    m: np.ndarray,
    d: np.ndarray,
    g: np.ndarray,
    step_size: float,
) -> np.ndarray:
    z_forward = z - step_size * gradient(m, d, g, z)
    z_projected = project_soc(z_forward)
    return (z - z_projected) / step_size


def solve_socp_pgd(
    m: np.ndarray,
    d: np.ndarray,
    g: np.ndarray,
    max_iters: int = 4000,
    tol: float = 1e-8,
    step_scale: float = 0.99,
) -> SOCPResult:
    """Solve SOCP with projected gradient descent (forward-backward splitting)."""
    validate_problem(m, d, g)

    if max_iters <= 0:
        raise ValueError("max_iters must be > 0.")
    if tol <= 0.0:
        raise ValueError("tol must be > 0.")
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1].")

    l_const = lipschitz_constant(m)
    step_size = step_scale / l_const

    n_dim = m.shape[1]
    z = np.zeros(n_dim, dtype=float)  # origin is always in SOC

    history: List[HistoryItem] = []
    pg_res0 = float(np.linalg.norm(projected_gradient_mapping(z, m, d, g, step_size)))
    history.append((0, objective(m, d, g, z), 0.0, pg_res0, soc_feasibility_gap(z)))

    converged = False

    for k in range(1, max_iters + 1):
        z_next = project_soc(z - step_size * gradient(m, d, g, z))

        update_norm = float(np.linalg.norm(z_next - z))
        obj = objective(m, d, g, z_next)
        pg_res = float(np.linalg.norm(projected_gradient_mapping(z_next, m, d, g, step_size)))
        feas_gap = soc_feasibility_gap(z_next)

        if not np.isfinite(obj):
            raise RuntimeError("Non-finite objective encountered during optimization.")

        history.append((k, obj, update_norm, pg_res, feas_gap))
        z = z_next

        if update_norm < tol and pg_res < 5.0 * tol:
            converged = True
            break

    return SOCPResult(
        z=z,
        history=history,
        converged=converged,
        iters_used=len(history) - 1,
        step_size=step_size,
    )


def make_synthetic_socp(
    seed: int = 2026,
    n_obs: int = 36,
    n_dim: int = 7,
    noise_std: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible SOCP instance with a known feasible reference point."""
    if n_dim < 2:
        raise ValueError("n_dim must be >= 2.")
    if n_obs < n_dim:
        raise ValueError("n_obs should be >= n_dim for a well-conditioned demo.")

    rng = np.random.default_rng(seed)

    m = rng.normal(size=(n_obs, n_dim))
    m[:n_dim, :] += 1.5 * np.eye(n_dim)

    u_ref = rng.normal(size=n_dim - 1)
    t_ref = float(np.linalg.norm(u_ref) + 0.8)
    z_ref = np.concatenate(([t_ref], u_ref))

    d = m @ z_ref + noise_std * rng.normal(size=n_obs)
    g = 0.08 * rng.normal(size=n_dim)

    return m, d, g, z_ref


def objective_monotone_check(history: Sequence[HistoryItem], tol: float = 1e-10) -> Tuple[bool, int]:
    violations = 0
    for i in range(1, len(history)):
        if history[i][1] > history[i - 1][1] + tol:
            violations += 1
    return violations == 0, violations


def print_history(history: Sequence[HistoryItem], max_lines: int = 10) -> None:
    print("iter | objective          | ||dz||            | pg_res            | feas_gap")
    print("----------------------------------------------------------------------------")

    shown = min(len(history), max_lines)
    for i in range(shown):
        it, obj, dz, pg_res, gap = history[i]
        print(f"{it:4d} | {obj:18.10e} | {dz:16.8e} | {pg_res:16.8e} | {gap:8.2e}")

    if len(history) > max_lines:
        omitted = len(history) - max_lines
        it, obj, dz, pg_res, gap = history[-1]
        print(f"... ({omitted} more iterations omitted)")
        print(f"{it:4d} | {obj:18.10e} | {dz:16.8e} | {pg_res:16.8e} | {gap:8.2e}  (last)")


def main() -> None:
    m, d, g, z_ref = make_synthetic_socp(seed=2026, n_obs=36, n_dim=7, noise_std=0.03)

    result = solve_socp_pgd(
        m=m,
        d=d,
        g=g,
        max_iters=4000,
        tol=1e-8,
        step_scale=0.99,
    )

    print("=== SOCP via Projected Gradient Descent ===")
    print(f"problem size: M shape = {m.shape}, cone dim = {m.shape[1]}")
    print(f"step size: {result.step_size:.6e}")
    print_history(result.history, max_lines=10)

    z_est = result.z
    final_obj = result.history[-1][1]
    final_pg_res = result.history[-1][3]
    final_gap = result.history[-1][4]
    monotone_ok, violations = objective_monotone_check(result.history)

    dist_to_ref = float(np.linalg.norm(z_est - z_ref))

    print("\n--- Summary ---")
    print(f"converged: {result.converged}")
    print(f"iterations used: {result.iters_used}")
    print(f"final objective: {final_obj:.10f}")
    print(f"final projected-gradient residual: {final_pg_res:.3e}")
    print(f"final SOC feasibility gap: {final_gap:.3e}")
    print(f"objective monotone check: {monotone_ok} (violations={violations})")
    print(f"distance to reference feasible point: {dist_to_ref:.6f}")
    print(f"reference point is feasible: {soc_feasibility_gap(z_ref) <= 1e-12}")
    print(f"estimated point is feasible: {soc_feasibility_gap(z_est) <= 1e-8}")

    print("\nreference z:", np.array2string(z_ref, precision=4, suppress_small=True))
    print("estimated z:", np.array2string(z_est, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
