"""Minimal runnable MVP: FEM eigenvalue problem on 1D interval.

Problem:
    -u'' = lambda * u, x in (0, 1), u(0)=u(1)=0

Discretization:
    P1 linear finite elements -> generalized symmetric eigenproblem
    K u_h = lambda_h M u_h
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.linalg import eigh as scipy_eigh  # type: ignore
except Exception:  # pragma: no cover - fallback is intentional
    scipy_eigh = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a deterministic refinement experiment."""

    n_elems_list: Tuple[int, ...] = (16, 32, 64, 128)
    num_modes: int = 4



def assemble_p1_matrices(n_elem: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global stiffness and mass matrices for 1D P1 FEM."""
    if n_elem < 2:
        raise ValueError("n_elem must be >= 2.")

    nodes = np.linspace(0.0, 1.0, n_elem + 1)
    stiffness = np.zeros((n_elem + 1, n_elem + 1), dtype=float)
    mass = np.zeros((n_elem + 1, n_elem + 1), dtype=float)

    for e in range(n_elem):
        x_l = nodes[e]
        x_r = nodes[e + 1]
        h = x_r - x_l

        k_local = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        m_local = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]], dtype=float)

        idx = (e, e + 1)
        stiffness[np.ix_(idx, idx)] += k_local
        mass[np.ix_(idx, idx)] += m_local

    return nodes, stiffness, mass



def solve_generalized_eigenproblem(
    stiffness: np.ndarray,
    mass: np.ndarray,
    num_modes: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Solve K v = lambda M v for smallest modes.

    Returns:
        eigenvalues, eigenvectors, solver_name
    """
    dof = stiffness.shape[0]
    if dof != mass.shape[0]:
        raise ValueError("stiffness/mass dimension mismatch")

    num_modes = min(num_modes, dof)
    if num_modes < 1:
        raise ValueError("num_modes must be >= 1")

    if scipy_eigh is not None:
        # Direct generalized Hermitian eigensolver for smallest subset.
        eigvals, eigvecs = scipy_eigh(
            stiffness,
            mass,
            subset_by_index=[0, num_modes - 1],
            check_finite=True,
        )
        solver_name = "scipy.linalg.eigh"
    else:
        # Fallback: reduce generalized EVP to standard symmetric EVP.
        # M = L L^T (Cholesky), then solve (L^{-1} K L^{-T}) y = lambda y, v=L^{-T}y.
        l_factor = np.linalg.cholesky(mass)
        reduced = np.linalg.solve(l_factor, np.linalg.solve(l_factor, stiffness).T)
        eigvals_all, eigvecs_reduced = np.linalg.eigh(reduced)
        eigvals = eigvals_all[:num_modes]
        eigvecs = np.linalg.solve(l_factor.T, eigvecs_reduced[:, :num_modes])
        solver_name = "numpy.linalg.eigh (Cholesky reduction)"

    # Sort defensively and M-normalize eigenvectors.
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    for j in range(eigvecs.shape[1]):
        v = eigvecs[:, j]
        mv = mass.dot(v)
        norm_m_sq = float(np.dot(v, mv))
        norm_m = math.sqrt(max(norm_m_sq, 0.0))
        if norm_m <= 0.0:
            raise RuntimeError("Encountered non-positive M-norm eigenvector")
        eigvecs[:, j] = v / norm_m

    return eigvals, eigvecs, solver_name



def exact_eigenvalues(num_modes: int) -> np.ndarray:
    """Exact eigenvalues for -u''=lambda*u on (0,1), Dirichlet boundaries."""
    ks = np.arange(1, num_modes + 1, dtype=float)
    return (ks * math.pi) ** 2



def run_refinement_study(config: ExperimentConfig) -> List[Dict[str, float]]:
    """Run FEM eigenvalue experiment over several mesh levels."""
    records: List[Dict[str, float]] = []

    for n_elem in config.n_elems_list:
        _nodes, stiffness, mass = assemble_p1_matrices(n_elem)

        # Dirichlet elimination: keep interior DoFs only.
        k_in = stiffness[1:-1, 1:-1]
        m_in = mass[1:-1, 1:-1]

        eigvals, _eigvecs, _solver_name = solve_generalized_eigenproblem(
            k_in,
            m_in,
            config.num_modes,
        )
        exact = exact_eigenvalues(eigvals.size)
        rel_err = np.abs(eigvals - exact) / exact

        record: Dict[str, float] = {
            "n_elem": float(n_elem),
            "h": float(1.0 / n_elem),
        }
        for i, lam in enumerate(eigvals, start=1):
            record[f"lambda{i}"] = float(lam)
            record[f"exact{i}"] = float(exact[i - 1])
            record[f"rel_err{i}"] = float(rel_err[i - 1])
        records.append(record)

    return records



def convergence_rates(records: List[Dict[str, float]], num_modes: int) -> List[Dict[str, float]]:
    """Compute empirical rates based on consecutive mesh halving."""
    rates: List[Dict[str, float]] = []

    for idx, rec in enumerate(records):
        row: Dict[str, float] = {"n_elem": rec["n_elem"], "h": rec["h"]}
        if idx == 0:
            for mode in range(1, num_modes + 1):
                row[f"rate{mode}"] = float("nan")
        else:
            prev = records[idx - 1]
            for mode in range(1, num_modes + 1):
                key = f"rel_err{mode}"
                if key in rec and key in prev and rec[key] > 0.0 and prev[key] > 0.0:
                    row[f"rate{mode}"] = math.log(prev[key] / rec[key], 2.0)
                else:
                    row[f"rate{mode}"] = float("nan")
        rates.append(row)

    return rates



def run_checks(records: List[Dict[str, float]], rates: List[Dict[str, float]], num_modes: int) -> None:
    """Basic numerical sanity checks for the MVP."""
    if not records:
        raise RuntimeError("No experiment records generated.")

    # Check positivity and ordering of eigenvalues at each mesh level.
    for rec in records:
        prev_lam = -1.0
        for mode in range(1, num_modes + 1):
            key = f"lambda{mode}"
            if key not in rec:
                continue
            lam = rec[key]
            if not math.isfinite(lam) or lam <= 0.0:
                raise RuntimeError(f"Non-positive or non-finite eigenvalue at mode {mode}: {lam}")
            if lam <= prev_lam:
                raise RuntimeError("Eigenvalues are not strictly increasing.")
            prev_lam = lam

    # First mode error should decrease with refinement.
    first_mode_errors = [rec["rel_err1"] for rec in records if "rel_err1" in rec]
    for i in range(1, len(first_mode_errors)):
        if not (first_mode_errors[i] < first_mode_errors[i - 1]):
            raise RuntimeError("First-mode relative error did not decrease monotonically.")

    # On the finest mesh, first-mode error should be small.
    if first_mode_errors[-1] >= 5e-4:
        raise RuntimeError(
            f"Finest first-mode relative error too large: {first_mode_errors[-1]:.3e}"
        )

    # Last observed convergence rate of first mode should approach second order.
    last_rate = rates[-1].get("rate1", float("nan"))
    if not math.isnan(last_rate) and last_rate < 1.8:
        raise RuntimeError(f"First-mode convergence rate too low: {last_rate:.4f}")



def print_report(records: List[Dict[str, float]], rates: List[Dict[str, float]], num_modes: int) -> None:
    """Print deterministic experiment summary."""
    print("FEM Eigenvalue MVP: -u'' = lambda u on (0,1), u(0)=u(1)=0")
    print("Discretization: P1 finite element, generalized EVP K u = lambda M u")
    print()

    header_cols = ["n_elem", "h"]
    for mode in range(1, num_modes + 1):
        header_cols.extend([f"lam{mode}", f"rel{mode}", f"rate{mode}"])
    print(" ".join(f"{c:>14s}" for c in header_cols))

    for rec, rate in zip(records, rates):
        line_items = [f"{int(rec['n_elem']):14d}", f"{rec['h']:14.6f}"]
        for mode in range(1, num_modes + 1):
            lam_key = f"lambda{mode}"
            rel_key = f"rel_err{mode}"
            rate_key = f"rate{mode}"

            lam_val = rec.get(lam_key, float("nan"))
            rel_val = rec.get(rel_key, float("nan"))
            rate_val = rate.get(rate_key, float("nan"))

            line_items.append(f"{lam_val:14.8f}")
            line_items.append(f"{rel_val:14.3e}")
            if math.isnan(rate_val):
                line_items.append(f"{'-':>14s}")
            else:
                line_items.append(f"{rate_val:14.6f}")

        print(" ".join(line_items))



def main() -> None:
    config = ExperimentConfig()
    records = run_refinement_study(config)
    rates = convergence_rates(records, config.num_modes)
    print_report(records, rates, config.num_modes)
    run_checks(records, rates, config.num_modes)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
