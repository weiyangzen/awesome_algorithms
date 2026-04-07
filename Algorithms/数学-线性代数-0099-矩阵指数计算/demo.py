"""Matrix exponential MVP via scaling-and-squaring with Padé(13)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from scipy.linalg import expm as scipy_expm
except Exception:  # pragma: no cover - scipy is optional for this MVP
    scipy_expm = None


THETA_13 = 5.371920351148152
PADE13_COEFFS = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)
EPS = 1e-15


@dataclass
class ExpmReport:
    name: str
    shape: tuple[int, int]
    norm_1: float
    scaling_steps: int
    inv_identity_error: float
    det_trace_relative_error: float
    exact_diff_fro: float | None
    scipy_diff_fro: float | None


def validate_square_matrix(a: np.ndarray, *, name: str = "A") -> None:
    """Validate finite square matrix input."""
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got ndim={a.ndim}")

    n, m = a.shape
    if n != m:
        raise ValueError(f"{name} must be square, got shape={a.shape}")

    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")


def pade13_approximant(a: np.ndarray) -> np.ndarray:
    """Return Padé(13) rational approximation to exp(a) (without scaling)."""
    b = PADE13_COEFFS
    identity = np.eye(a.shape[0], dtype=a.dtype)

    a2 = a @ a
    a4 = a2 @ a2
    a6 = a4 @ a2

    u = a @ (
        a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2)
        + b[7] * a6
        + b[5] * a4
        + b[3] * a2
        + b[1] * identity
    )
    v = (
        a6 @ (b[12] * a6 + b[10] * a4 + b[8] * a2)
        + b[6] * a6
        + b[4] * a4
        + b[2] * a2
        + b[0] * identity
    )

    # (V-U)^{-1}(V+U), solved as linear system to avoid explicit inverse.
    return np.linalg.solve(v - u, v + u)


def matrix_exponential_pade13(a: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Compute exp(a) using scaling-and-squaring with Padé(13)."""
    validate_square_matrix(a, name="A")

    norm_1 = float(np.linalg.norm(a, ord=1))
    if norm_1 == 0.0:
        return np.eye(a.shape[0], dtype=a.dtype), 0, norm_1

    scaling_steps = 0
    if norm_1 > THETA_13:
        scaling_steps = int(math.ceil(math.log2(norm_1 / THETA_13)))

    a_scaled = a / (2**scaling_steps)
    exp_a = pade13_approximant(a_scaled)

    for _ in range(scaling_steps):
        exp_a = exp_a @ exp_a

    return exp_a, scaling_steps, norm_1


def analyze_case(name: str, a: np.ndarray, expected: np.ndarray | None = None) -> ExpmReport:
    """Run one test case and collect diagnostics."""
    exp_a, scaling_steps, norm_1 = matrix_exponential_pade13(a)
    inv_identity_error = float(
        np.linalg.norm(exp_a @ matrix_exponential_pade13(-a)[0] - np.eye(a.shape[0]), ord=np.inf)
    )

    det_lhs = np.linalg.det(exp_a)
    det_rhs = np.exp(np.trace(a))
    det_trace_relative_error = float(abs(det_lhs - det_rhs) / max(abs(det_rhs), 1.0))

    exact_diff_fro = None
    if expected is not None:
        exact_diff_fro = float(np.linalg.norm(exp_a - expected, ord="fro"))

    scipy_diff_fro = None
    if scipy_expm is not None:
        scipy_diff_fro = float(np.linalg.norm(exp_a - scipy_expm(a), ord="fro"))

    return ExpmReport(
        name=name,
        shape=a.shape,
        norm_1=norm_1,
        scaling_steps=scaling_steps,
        inv_identity_error=inv_identity_error,
        det_trace_relative_error=det_trace_relative_error,
        exact_diff_fro=exact_diff_fro,
        scipy_diff_fro=scipy_diff_fro,
    )


def run_checks(report: ExpmReport) -> None:
    """Assert numerical quality thresholds for one report."""
    if report.inv_identity_error >= 1e-10:
        raise AssertionError(
            f"{report.name}: exp(A)exp(-A)-I too large: {report.inv_identity_error:.3e}"
        )

    if report.det_trace_relative_error >= 1e-10:
        raise AssertionError(
            f"{report.name}: det/trace identity error too large: "
            f"{report.det_trace_relative_error:.3e}"
        )

    if report.exact_diff_fro is not None and report.exact_diff_fro >= 1e-11:
        raise AssertionError(
            f"{report.name}: mismatch vs closed-form expected value: {report.exact_diff_fro:.3e}"
        )

    if report.scipy_diff_fro is not None and report.scipy_diff_fro >= 1e-10:
        raise AssertionError(
            f"{report.name}: mismatch vs scipy.linalg.expm too large: {report.scipy_diff_fro:.3e}"
        )


def build_demo_cases() -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Create deterministic test cases and optional closed-form references."""
    zero = np.zeros((3, 3), dtype=np.float64)
    zero_expected = np.eye(3, dtype=np.float64)

    diag_values = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    diagonal = np.diag(diag_values)
    diagonal_expected = np.diag(np.exp(diag_values))

    # Rotation generator J = [[0,-1],[1,0]], exp(J) is rotation by 1 radian.
    rotation_generator = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    c = math.cos(1.0)
    s = math.sin(1.0)
    rotation_expected = np.array([[c, -s], [s, c]], dtype=np.float64)

    # Nilpotent N with N^3 = 0 => exp(N) = I + N + N^2/2.
    nilpotent = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    n2 = nilpotent @ nilpotent
    nilpotent_expected = np.eye(3, dtype=np.float64) + nilpotent + 0.5 * n2

    rng = np.random.default_rng(2026)
    random_dense = rng.normal(size=(4, 4)).astype(np.float64)

    return [
        ("zero_3x3", zero, zero_expected),
        ("diagonal_3x3", diagonal, diagonal_expected),
        ("rotation_generator_2x2", rotation_generator, rotation_expected),
        ("nilpotent_3x3", nilpotent, nilpotent_expected),
        ("random_dense_4x4", random_dense, None),
    ]


def main() -> None:
    print("Matrix exponential MVP (Scaling & Squaring + Padé(13))")
    print("=" * 72)

    for name, matrix, expected in build_demo_cases():
        report = analyze_case(name, matrix, expected=expected)
        run_checks(report)

        print(f"Case: {report.name}")
        print(f"  shape                    : {report.shape}")
        print(f"  ||A||_1                  : {report.norm_1:.6e}")
        print(f"  scaling steps (s)        : {report.scaling_steps}")
        print(f"  ||exp(A)exp(-A)-I||_inf  : {report.inv_identity_error:.6e}")
        print(f"  det-trace relative error : {report.det_trace_relative_error:.6e}")

        if report.exact_diff_fro is None:
            print("  closed-form diff (fro)   : n/a")
        else:
            print(f"  closed-form diff (fro)   : {report.exact_diff_fro:.6e}")

        if report.scipy_diff_fro is None:
            print("  scipy diff (fro)         : skipped (scipy not available)")
        else:
            print(f"  scipy diff (fro)         : {report.scipy_diff_fro:.6e}")

        print("-" * 72)

    print("All checks passed.")


if __name__ == "__main__":
    main()
