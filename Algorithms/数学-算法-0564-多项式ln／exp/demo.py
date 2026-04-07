"""Polynomial ln/exp over truncated formal power series.

This MVP implements:
- ln(A): requires A(0)=1
- exp(F): requires F(0)=0
with coefficients truncated modulo x^n.
"""

from __future__ import annotations

from time import perf_counter
from typing import List, Sequence

import numpy as np

EPS = 1e-12


def poly_truncate(poly: Sequence[float], n: int) -> List[float]:
    if n <= 0:
        return []
    out = list(poly[:n])
    if len(out) < n:
        out.extend([0.0] * (n - len(out)))
    return out


def poly_mul_trunc(a: Sequence[float], b: Sequence[float], n: int) -> List[float]:
    """Naive convolution truncated to first n coefficients."""
    if n <= 0:
        return []
    out = [0.0] * n
    la, lb = len(a), len(b)
    for i in range(min(la, n)):
        ai = a[i]
        if abs(ai) < EPS:
            continue
        upper = min(lb, n - i)
        for j in range(upper):
            out[i + j] += ai * b[j]
    return out


def poly_derivative(a: Sequence[float]) -> List[float]:
    if len(a) <= 1:
        return [0.0]
    return [i * a[i] for i in range(1, len(a))]


def poly_integral(a: Sequence[float], n: int) -> List[float]:
    """Integral with zero constant term, truncated to n coefficients."""
    out = [0.0] * max(n, 1)
    for i in range(1, n):
        if i - 1 < len(a):
            out[i] = a[i - 1] / i
    return out[:n]


def poly_inv(a: Sequence[float], n: int) -> List[float]:
    """Newton iteration for inverse series: B = A^{-1} mod x^n."""
    if n <= 0:
        return []
    if not a or abs(a[0]) < EPS:
        raise ValueError("poly_inv requires non-zero constant term")

    g = [1.0 / a[0]]
    m = 1
    while m < n:
        m2 = min(2 * m, n)
        ag = poly_mul_trunc(a, g, m2)

        two_minus_ag = [0.0] * m2
        two_minus_ag[0] = 2.0 - ag[0]
        for i in range(1, m2):
            two_minus_ag[i] = -ag[i]

        g = poly_mul_trunc(g, two_minus_ag, m2)
        m = m2

    return g[:n]


def poly_ln(a: Sequence[float], n: int) -> List[float]:
    """ln(A) = integral(A'/A), requires A(0)=1, truncated to x^n."""
    if n <= 0:
        return []
    if not a:
        raise ValueError("poly_ln requires non-empty polynomial")
    if abs(a[0] - 1.0) > 1e-9:
        raise ValueError("poly_ln requires constant term a0=1")

    da = poly_derivative(a)
    inv_a = poly_inv(a, n)
    quot = poly_mul_trunc(da, inv_a, max(n - 1, 0))
    return poly_integral(quot, n)


def poly_exp(f: Sequence[float], n: int) -> List[float]:
    """Newton iteration for exp(F), requires F(0)=0, truncated to x^n."""
    if n <= 0:
        return []
    if f and abs(f[0]) > 1e-9:
        raise ValueError("poly_exp requires constant term f0=0")

    g = [1.0]  # exp(0) = 1
    m = 1
    while m < n:
        m2 = min(2 * m, n)
        ln_g = poly_ln(g, m2)

        delta = [0.0] * m2
        for i in range(m2):
            fi = f[i] if i < len(f) else 0.0
            delta[i] = fi - ln_g[i]
        delta[0] += 1.0  # 1 - ln(g) + f

        g = poly_mul_trunc(g, delta, m2)
        m = m2

    return g[:n]


def poly_eval(poly: Sequence[float], x: float) -> float:
    acc = 0.0
    for c in reversed(poly):
        acc = acc * x + c
    return acc


def max_abs_diff(a: Sequence[float], b: Sequence[float], n: int) -> float:
    diff = 0.0
    for i in range(n):
        ai = a[i] if i < len(a) else 0.0
        bi = b[i] if i < len(b) else 0.0
        diff = max(diff, abs(ai - bi))
    return diff


def main() -> None:
    n = 12

    # f(0)=0 for exp(f)
    f = [0.0, 0.35, -0.20, 0.10, -0.03, 0.015, 0.0, -0.008, 0.003]

    # a(0)=1 for ln(a)
    a = [1.0, 0.50, -0.25, 0.18, -0.07, 0.03, -0.015, 0.005]

    t0 = perf_counter()
    exp_f = poly_exp(f, n)
    ln_exp_f = poly_ln(exp_f, n)
    t1 = perf_counter()

    ln_a = poly_ln(a, n)
    exp_ln_a = poly_exp(ln_a, n)
    t2 = perf_counter()

    err_ln_exp = max_abs_diff(ln_exp_f, poly_truncate(f, n), n)
    err_exp_ln = max_abs_diff(exp_ln_a, poly_truncate(a, n), n)

    # Numeric sanity check at small x: exp(f(x)) vs polynomial exp(f) evaluated at x.
    xs = np.linspace(-0.15, 0.15, 7)
    numeric_err = 0.0
    for x in xs:
        fx = poly_eval(f, float(x))
        approx = poly_eval(exp_f, float(x))
        truth = float(np.exp(fx))
        numeric_err = max(numeric_err, abs(approx - truth))

    print("=== Polynomial ln/exp MVP ===")
    print(f"Truncation order n: {n}")
    print(f"Max coeff error |ln(exp(f)) - f|: {err_ln_exp:.3e}")
    print(f"Max coeff error |exp(ln(a)) - a|: {err_exp_ln:.3e}")
    print(f"Max numeric error on x in [-0.15, 0.15]: {numeric_err:.3e}")
    print(f"Time for exp->ln test: {(t1 - t0) * 1e3:.3f} ms")
    print(f"Time for ln->exp test: {(t2 - t1) * 1e3:.3f} ms")

    print("\nFirst 8 coeffs of exp(f):")
    print("  " + ", ".join(f"{c:+.8f}" for c in exp_f[:8]))

    print("\nFirst 8 coeffs of ln(a):")
    print("  " + ", ".join(f"{c:+.8f}" for c in ln_a[:8]))

    ok = err_ln_exp < 1e-8 and err_exp_ln < 1e-8 and numeric_err < 5e-6
    print(f"\nValidation: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
