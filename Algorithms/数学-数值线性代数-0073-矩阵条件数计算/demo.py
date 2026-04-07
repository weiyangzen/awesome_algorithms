"""Minimal runnable MVP for matrix condition number computation.

This demo computes condition numbers in multiple norms without using
`numpy.linalg.cond` as the primary implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ConditionNumberReport:
    """Container for condition-number diagnostics of a square matrix."""

    n: int
    rank: int
    sigma_max: float
    sigma_min: float
    cond_2: float
    cond_1: float
    cond_inf: float
    cond_fro: float
    reciprocal_cond_2: float
    singular: bool


@dataclass
class PerturbationReport:
    """Observed and predicted error amplification under RHS perturbation."""

    rel_rhs_perturb: float
    rel_solution_change: float
    bound_by_kappa2: float
    amplification_factor: float


def _as_finite_2d_array(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("matrix must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError("matrix contains non-finite values")
    return arr


def _require_square(matrix: np.ndarray) -> np.ndarray:
    arr = _as_finite_2d_array(matrix)
    m, n = arr.shape
    if m != n:
        raise ValueError(f"matrix must be square, got shape=({m}, {n})")
    return arr


def _effective_rcond(n: int, rcond: float | None) -> float:
    if rcond is None:
        return np.finfo(float).eps * max(1, n)
    if rcond <= 0 or not np.isfinite(rcond):
        raise ValueError("rcond must be a positive finite number")
    return float(rcond)


def _orthogonal_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    return Q


def build_matrix_with_target_cond(n: int, target_cond: float, seed: int = 0) -> np.ndarray:
    """Build a full-rank matrix with approximately controlled 2-norm condition number."""

    if n <= 0:
        raise ValueError("n must be positive")
    if target_cond < 1.0 or not np.isfinite(target_cond):
        raise ValueError("target_cond must be finite and >= 1")

    U = _orthogonal_matrix(n, seed=seed)
    V = _orthogonal_matrix(n, seed=seed + 1)
    singular_values = np.geomspace(1.0, 1.0 / target_cond, num=n)
    # Scale U's columns directly instead of forming an explicit diagonal matrix.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        A = (U * singular_values) @ V.T
    if not np.isfinite(A).all():
        raise FloatingPointError("constructed matrix contains non-finite values")
    return A


def hilbert_matrix(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.arange(1, n + 1, dtype=float)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)


def compute_condition_report(matrix: np.ndarray, rcond: float | None = None) -> ConditionNumberReport:
    """Compute condition numbers in 1/2/inf/F norms for a square matrix.

    Definition used for non-singular A:
    kappa_p(A) = ||A||_p * ||A^{-1}||_p

    The 2-norm condition number is also sigma_max / sigma_min.
    """

    A = _require_square(matrix)
    n = A.shape[0]
    tol = _effective_rcond(n, rcond)

    singular_values = np.linalg.svd(A, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])

    threshold = tol * sigma_max
    singular = sigma_max == 0.0 or sigma_min <= threshold

    rank = int(np.sum(singular_values > threshold))

    if singular:
        cond_2 = float("inf")
        cond_1 = float("inf")
        cond_inf = float("inf")
        cond_fro = float("inf")
        rcond_2 = 0.0
    else:
        cond_2 = sigma_max / sigma_min
        A_inv = np.linalg.solve(A, np.eye(n))
        cond_1 = float(np.linalg.norm(A, ord=1) * np.linalg.norm(A_inv, ord=1))
        cond_inf = float(np.linalg.norm(A, ord=np.inf) * np.linalg.norm(A_inv, ord=np.inf))
        cond_fro = float(np.linalg.norm(A, ord="fro") * np.linalg.norm(A_inv, ord="fro"))
        rcond_2 = 1.0 / cond_2

    return ConditionNumberReport(
        n=n,
        rank=rank,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        cond_2=float(cond_2),
        cond_1=float(cond_1),
        cond_inf=float(cond_inf),
        cond_fro=float(cond_fro),
        reciprocal_cond_2=float(rcond_2),
        singular=singular,
    )


def rhs_perturbation_experiment(
    matrix: np.ndarray,
    cond_2: float,
    perturb_scale: float = 1e-7,
    seed: int = 42,
) -> PerturbationReport:
    """Measure sensitivity of Ax=b to a small perturbation on b."""

    A = _require_square(matrix)
    if not np.isfinite(cond_2):
        raise ValueError("cond_2 must be finite for perturbation experiment")
    if perturb_scale <= 0 or not np.isfinite(perturb_scale):
        raise ValueError("perturb_scale must be a positive finite number")

    rng = np.random.default_rng(seed)
    n = A.shape[0]

    b = rng.standard_normal(n)
    db = rng.standard_normal(n)
    db *= perturb_scale / (np.linalg.norm(db) + 1e-15)

    x = np.linalg.solve(A, b)
    x_perturbed = np.linalg.solve(A, b + db)

    rel_rhs = float(np.linalg.norm(db) / (np.linalg.norm(b) + 1e-15))
    rel_sol = float(np.linalg.norm(x_perturbed - x) / (np.linalg.norm(x) + 1e-15))
    bound = float(cond_2 * rel_rhs)
    amplification = float(rel_sol / (rel_rhs + 1e-15))

    return PerturbationReport(
        rel_rhs_perturb=rel_rhs,
        rel_solution_change=rel_sol,
        bound_by_kappa2=bound,
        amplification_factor=amplification,
    )


def _fmt(x: float) -> str:
    if np.isinf(x):
        return "inf"
    return f"{x:.6e}"


def run_case(name: str, matrix: np.ndarray) -> Dict[str, float]:
    report = compute_condition_report(matrix)

    cond2_np = float(np.linalg.cond(matrix, p=2))
    cond2_gap = abs(report.cond_2 - cond2_np)

    print(f"\nCase: {name}")
    print("-" * 72)
    print(f"shape                          : {matrix.shape}")
    print(f"rank / n                       : {report.rank} / {report.n}")
    print(f"sigma_max                      : {_fmt(report.sigma_max)}")
    print(f"sigma_min                      : {_fmt(report.sigma_min)}")
    print(f"kappa_2 (manual SVD)           : {_fmt(report.cond_2)}")
    print(f"kappa_2 (np.linalg.cond check) : {_fmt(cond2_np)}")
    print(f"|difference|                   : {_fmt(cond2_gap)}")
    print(f"kappa_1                        : {_fmt(report.cond_1)}")
    print(f"kappa_inf                      : {_fmt(report.cond_inf)}")
    print(f"kappa_F                        : {_fmt(report.cond_fro)}")
    print(f"reciprocal kappa_2             : {_fmt(report.reciprocal_cond_2)}")
    print(f"singular flag                  : {report.singular}")

    perturb_result = None
    if not report.singular:
        perturb_result = rhs_perturbation_experiment(matrix, cond_2=report.cond_2)
        print("\nRHS perturbation experiment (Ax=b):")
        print(f"relative perturbation in b      : {_fmt(perturb_result.rel_rhs_perturb)}")
        print(f"relative change in solution     : {_fmt(perturb_result.rel_solution_change)}")
        print(f"kappa_2 * rel_b (upper bound)   : {_fmt(perturb_result.bound_by_kappa2)}")
        print(f"observed amplification factor   : {_fmt(perturb_result.amplification_factor)}")

    return {
        "cond2_manual": report.cond_2,
        "cond2_numpy": cond2_np,
        "cond2_gap": cond2_gap,
        "rel_rhs": perturb_result.rel_rhs_perturb if perturb_result else float("nan"),
        "rel_sol": perturb_result.rel_solution_change if perturb_result else float("nan"),
    }


def main() -> None:
    matrix_well = build_matrix_with_target_cond(n=8, target_cond=25.0, seed=2026)
    matrix_ill = build_matrix_with_target_cond(n=8, target_cond=1e8, seed=2027)
    matrix_hilbert = hilbert_matrix(10)

    print("Matrix Condition Number MVP")
    print("=" * 72)
    print("Definition: kappa_p(A) = ||A||_p * ||A^{-1}||_p (non-singular square A)")

    run_case("controlled cond ~ 2.5e1", matrix_well)
    run_case("controlled cond ~ 1e8", matrix_ill)
    run_case("Hilbert(10)", matrix_hilbert)


if __name__ == "__main__":
    main()
