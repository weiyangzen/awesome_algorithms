"""Minimal runnable MVP for Muller method root finding.

The implementation keeps the core iteration explicit and traceable:
- No black-box root solver is used in the main algorithm.
- `numpy.roots` is only used as a reference for polynomial test cases.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


Number = complex


@dataclass
class MullerResult:
    root: Number
    f_at_root: Number
    iterations: int
    converged: bool
    history: List[Dict[str, Number | float | int]]


def muller(
    func: Callable[[Number], Number],
    x0: Number,
    x1: Number,
    x2: Number,
    tol: float = 1e-12,
    ftol: float = 1e-12,
    max_iter: int = 50,
) -> MullerResult:
    """Find one root of f(x)=0 using Muller's method.

    Parameters
    ----------
    func:
        Scalar function in real/complex domain.
    x0, x1, x2:
        Three initial guesses.
    tol:
        Tolerance for step size.
    ftol:
        Tolerance for residual magnitude.
    max_iter:
        Max number of iterations.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if tol <= 0 or ftol <= 0:
        raise ValueError("tol and ftol must be positive")

    x0 = complex(x0)
    x1 = complex(x1)
    x2 = complex(x2)
    f0 = complex(func(x0))
    f1 = complex(func(x1))
    f2 = complex(func(x2))

    history: List[Dict[str, Number | float | int]] = []

    for k in range(1, max_iter + 1):
        h1 = x1 - x0
        h2 = x2 - x1

        if abs(h1) == 0.0 or abs(h2) == 0.0:
            raise ValueError("Initial/current points must be distinct for Muller method")

        delta1 = (f1 - f0) / h1
        delta2 = (f2 - f1) / h2
        hsum = h1 + h2
        if abs(hsum) == 0.0:
            raise ValueError("Degenerate geometry: h1 + h2 == 0")

        d = (delta2 - delta1) / hsum
        b = delta2 + h2 * d
        rad = cmath.sqrt(b * b - 4.0 * f2 * d)

        # Choose the denominator with larger magnitude to reduce cancellation.
        denom_plus = b + rad
        denom_minus = b - rad
        denom = denom_plus if abs(denom_plus) >= abs(denom_minus) else denom_minus

        if abs(denom) == 0.0:
            raise RuntimeError("Numerical breakdown: denominator is zero")

        step = -2.0 * f2 / denom
        x3 = x2 + step
        f3 = complex(func(x3))

        record: Dict[str, Number | float | int] = {
            "iter": k,
            "x": x3,
            "fx": f3,
            "step_abs": abs(step),
            "fx_abs": abs(f3),
        }
        history.append(record)

        if abs(step) <= tol * (1.0 + abs(x3)) or abs(f3) <= ftol:
            return MullerResult(
                root=x3,
                f_at_root=f3,
                iterations=k,
                converged=True,
                history=history,
            )

        x0, x1, x2 = x1, x2, x3
        f0, f1, f2 = f1, f2, f3

    return MullerResult(
        root=x2,
        f_at_root=f2,
        iterations=max_iter,
        converged=False,
        history=history,
    )


def polynomial_function(coefficients: Sequence[float | complex]) -> Callable[[Number], Number]:
    coeffs = np.asarray(coefficients, dtype=np.complex128)

    def _f(x: Number) -> Number:
        return complex(np.polyval(coeffs, x))

    return _f


def nearest_reference_root(
    coefficients: Sequence[float | complex], approx_root: Number
) -> Tuple[Number, np.ndarray]:
    roots = np.roots(np.asarray(coefficients, dtype=np.complex128))
    idx = int(np.argmin(np.abs(roots - approx_root)))
    return complex(roots[idx]), roots


def format_complex(z: Number, digits: int = 12) -> str:
    if abs(z.imag) < 1e-14:
        return f"{z.real:.{digits}e}"
    return f"{z.real:.{digits}e} {z.imag:+.{digits}e}j"


def print_case_result(
    name: str,
    coefficients: Sequence[float | complex],
    initial_triplet: Tuple[Number, Number, Number],
    tol: float,
    ftol: float,
    max_iter: int,
    trace_limit: int = 6,
) -> None:
    func = polynomial_function(coefficients)
    result = muller(
        func=func,
        x0=initial_triplet[0],
        x1=initial_triplet[1],
        x2=initial_triplet[2],
        tol=tol,
        ftol=ftol,
        max_iter=max_iter,
    )

    reference, all_roots = nearest_reference_root(coefficients, result.root)
    abs_err = abs(result.root - reference)

    print(f"\n=== {name} ===")
    print(f"coefficients: {list(coefficients)}")
    print(
        "initial guesses: "
        f"[{format_complex(initial_triplet[0])}, {format_complex(initial_triplet[1])}, {format_complex(initial_triplet[2])}]"
    )
    print(f"converged: {result.converged}")
    print(f"iterations: {result.iterations}")
    print(f"approx_root: {format_complex(result.root)}")
    print(f"f(approx_root): {format_complex(result.f_at_root)}")
    print(f"reference_root(nearest): {format_complex(reference)}")
    print(f"abs_error_to_reference: {abs_err:.6e}")

    roots_str = ", ".join(format_complex(complex(r), digits=6) for r in all_roots)
    print(f"all polynomial roots (numpy.roots): [{roots_str}]")

    print("trace (first iterations):")
    for item in result.history[:trace_limit]:
        xk = format_complex(item["x"])  # type: ignore[index]
        fxk = format_complex(item["fx"])  # type: ignore[index]
        step_abs = float(item["step_abs"])  # type: ignore[arg-type]
        fx_abs = float(item["fx_abs"])  # type: ignore[arg-type]
        it = int(item["iter"])  # type: ignore[arg-type]
        print(
            f"  iter={it:02d}, x={xk}, |step|={step_abs:.3e}, "
            f"|f(x)|={fx_abs:.3e}, f(x)={fxk}"
        )


def main() -> None:
    cases = [
        {
            "name": "Case 1: x^3 - x - 2 = 0 (real root)",
            "coefficients": [1.0, 0.0, -1.0, -2.0],
            "initial_triplet": (0.0, 1.0, 2.0),
        },
        {
            "name": "Case 2: x^4 + 1 = 0 (complex root)",
            "coefficients": [1.0, 0.0, 0.0, 0.0, 1.0],
            "initial_triplet": (0.0 + 0.0j, 0.5 + 0.2j, 1.0 + 1.0j),
        },
        {
            "name": "Case 3: x^3 - 6x^2 + 11x - 6 = 0 (multiple real roots)",
            "coefficients": [1.0, -6.0, 11.0, -6.0],
            "initial_triplet": (1.5, 2.2, 2.8),
        },
    ]

    tol = 1e-12
    ftol = 1e-12
    max_iter = 50

    print("Muller Method MVP")
    print(f"settings: tol={tol}, ftol={ftol}, max_iter={max_iter}")

    for case in cases:
        print_case_result(
            name=case["name"],
            coefficients=case["coefficients"],
            initial_triplet=case["initial_triplet"],
            tol=tol,
            ftol=ftol,
            max_iter=max_iter,
            trace_limit=8,
        )


if __name__ == "__main__":
    main()
