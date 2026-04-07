"""Minimal runnable MVP for Power Iteration (CS-0324).

Goal:
    Estimate the dominant eigenvalue/eigenvector of a symmetric matrix
    using a source-visible power iteration implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PowerIterationConfig:
    """Configuration for the power-iteration MVP."""

    dimension: int = 8
    seed: int = 2026
    max_iters: int = 500
    tol: float = 1e-12
    residual_tol: float = 1e-10

    def validate(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be >= 2")
        if self.max_iters < 2:
            raise ValueError("max_iters must be >= 2")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")
        if self.residual_tol <= 0.0:
            raise ValueError("residual_tol must be positive")


def build_symmetric_matrix_with_known_spectrum(
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build A = Q * diag(eigs) * Q^T with controllable dominant eigenvalue."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(raw)

    dominant = float(n + 4)
    trailing = np.linspace(3.0, 0.5, n - 1, dtype=np.float64)
    eigvals = np.concatenate(([dominant], trailing))

    a = q @ np.diag(eigvals) @ q.T
    a = 0.5 * (a + a.T)
    dominant_vec = q[:, 0]
    return a, eigvals, dominant_vec


def rayleigh_quotient(a: np.ndarray, x: np.ndarray) -> float:
    """Return x^T A x / x^T x."""
    ax = a @ x
    denom = float(np.dot(x, x))
    if denom == 0.0:
        raise ValueError("zero vector encountered in Rayleigh quotient")
    return float(np.dot(x, ax) / denom)


def residual_norm(a: np.ndarray, x: np.ndarray, eigenvalue: float) -> float:
    """Return ||A x - lambda x||_2."""
    r = a @ x - eigenvalue * x
    return float(np.linalg.norm(r))


def power_iteration(
    a: np.ndarray,
    seed: int,
    max_iters: int,
    tol: float,
    residual_tol: float,
) -> tuple[float, np.ndarray, list[dict[str, float]], int, bool]:
    """Run power iteration and return estimate, vector, history, iterations, converged."""
    n = a.shape[0]
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(n)
    x_norm = float(np.linalg.norm(x))
    if x_norm == 0.0:
        raise ValueError("random initialization produced a zero vector")
    x = x / x_norm

    history: list[dict[str, float]] = []
    prev_lambda: float | None = None
    converged = False

    for iteration in range(1, max_iters + 1):
        y = a @ x
        y_norm = float(np.linalg.norm(y))
        if y_norm == 0.0:
            raise ValueError("A @ x became a zero vector; cannot continue")
        x = y / y_norm

        lambda_est = rayleigh_quotient(a, x)
        res = residual_norm(a, x, lambda_est)
        delta = np.inf if prev_lambda is None else abs(lambda_est - prev_lambda)

        history.append(
            {
                "iter": float(iteration),
                "lambda_est": lambda_est,
                "delta_lambda": float(delta),
                "residual_2norm": res,
            }
        )

        if prev_lambda is not None and delta < tol and res < residual_tol:
            converged = True
            break
        prev_lambda = lambda_est

    return lambda_est, x, history, iteration, converged


def main() -> None:
    cfg = PowerIterationConfig()
    cfg.validate()

    a, expected_eigs, expected_vec = build_symmetric_matrix_with_known_spectrum(
        n=cfg.dimension,
        seed=cfg.seed,
    )

    lambda_est, vec_est, history, iterations, converged = power_iteration(
        a=a,
        seed=cfg.seed + 1,
        max_iters=cfg.max_iters,
        tol=cfg.tol,
        residual_tol=cfg.residual_tol,
    )

    evals_ref, evecs_ref = np.linalg.eigh(a)
    idx = int(np.argmax(evals_ref))
    lambda_ref = float(evals_ref[idx])
    vec_ref = evecs_ref[:, idx]

    value_error = abs(lambda_est - lambda_ref)
    final_residual = float(history[-1]["residual_2norm"])
    cos_alignment = abs(float(np.dot(vec_est, vec_ref)))
    known_value_error = abs(lambda_est - float(expected_eigs[0]))
    known_vec_alignment = abs(float(np.dot(vec_est, expected_vec)))

    hist_df = pd.DataFrame(history)
    tail_rows = min(8, len(hist_df))

    print("=== Power Iteration MVP ===")
    print(
        f"dimension={cfg.dimension}, max_iters={cfg.max_iters}, "
        f"tol={cfg.tol:.1e}, residual_tol={cfg.residual_tol:.1e}"
    )
    print(f"iterations_used={iterations}, converged={converged}")
    print("Recent iteration history:")
    print(hist_df.tail(tail_rows).to_string(index=False, float_format=lambda x: f"{x:.12e}"))

    print("\nFinal summary:")
    print(f"lambda_est={lambda_est:.12e}")
    print(f"lambda_ref={lambda_ref:.12e}")
    print(f"|lambda_est-lambda_ref|={value_error:.12e}")
    print(f"|lambda_est-lambda_known|={known_value_error:.12e}")
    print(f"residual_2norm={final_residual:.12e}")
    print(f"|cos(vec_est, vec_ref)|={cos_alignment:.12e}")
    print(f"|cos(vec_est, vec_known)|={known_vec_alignment:.12e}")

    assert converged, "Power iteration did not converge within max_iters"
    assert value_error < 1e-10, f"Eigenvalue mismatch too large: {value_error:.3e}"
    assert final_residual < 1e-10, f"Residual too large: {final_residual:.3e}"
    assert cos_alignment > 1.0 - 1e-9, f"Vector alignment too small: {cos_alignment:.12f}"

    print("Validation: PASS")


if __name__ == "__main__":
    main()
