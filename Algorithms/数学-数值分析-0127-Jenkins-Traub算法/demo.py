"""Minimal runnable MVP for Jenkins-Traub algorithm (MATH-0127).

This is a Jenkins-Traub-style educational implementation:
- Stage 1: no-shift initialization of K polynomial.
- Stage 2: fixed-shift K updates.
- Stage 3: variable-shift root refinement.

It is intentionally small and transparent, not a production-grade RPOLY clone.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def trim_leading_zeros(coeffs: Sequence[complex], tol: float = 1e-14) -> np.ndarray:
    """Remove near-zero leading coefficients and return complex ndarray."""
    arr = np.asarray(coeffs, dtype=complex)
    idx = 0
    while idx < arr.size - 1 and abs(arr[idx]) <= tol:
        idx += 1
    return arr[idx:]


def horner_eval(coeffs: Sequence[complex], z: complex) -> complex:
    """Evaluate polynomial (descending coefficients) at z using Horner's rule."""
    value = 0.0 + 0.0j
    for c in coeffs:
        value = value * z + c
    return value


def cauchy_root_bound(coeffs: Sequence[complex]) -> float:
    """Simple Cauchy bound: all roots satisfy |z| <= 1 + max |a_i/a_0|."""
    p = trim_leading_zeros(coeffs)
    if p.size <= 1:
        return 0.0
    a0 = p[0]
    tail = np.abs(p[1:] / a0)
    if tail.size == 0:
        return 0.0
    return float(1.0 + np.max(tail))


def synthetic_division(coeffs: Sequence[complex], root: complex) -> Tuple[np.ndarray, complex]:
    """Divide polynomial by (x - root) using synthetic division.

    Returns (quotient_coeffs, remainder).
    """
    p = np.asarray(coeffs, dtype=complex)
    n = p.size - 1
    if n < 1:
        raise ValueError("Polynomial degree must be >= 1 for synthetic division")

    q = np.empty(n, dtype=complex)
    q[0] = p[0]
    for i in range(1, n):
        q[i] = p[i] + q[i - 1] * root
    rem = p[-1] + q[-1] * root
    return q, rem


def jt_k_step(p: np.ndarray, k: np.ndarray, s: complex, eps: float = 1e-14) -> np.ndarray:
    """One Jenkins-Traub-style K update at shift s.

    Uses a practical recurrence form:
        N(x) = (x - s)K(x) - beta * P(x), beta = K(s)/P(s)
        K_next(x) = N(x) / (x - s)

    This keeps degree(K) = degree(P)-1.
    """
    p_s = horner_eval(p, s)
    if abs(p_s) <= eps:
        return k

    k_s = horner_eval(k, s)
    if abs(k_s) <= eps:
        fallback = np.polyder(p)
        if fallback.size == 0:
            return np.array([1.0 + 0.0j], dtype=complex)
        return np.asarray(fallback, dtype=complex)

    beta = k_s / p_s

    # (x - s)K(x)
    numerator = np.polysub(np.polymul(k, np.array([1.0 + 0.0j, -s], dtype=complex)), beta * p)
    q, _ = np.polydiv(numerator, np.array([1.0 + 0.0j, -s], dtype=complex))
    q = trim_leading_zeros(q)
    if q.size == 0:
        return np.array([1.0 + 0.0j], dtype=complex)
    return q


def newton_polish(p: np.ndarray, z0: complex, max_iter: int = 20, tol: float = 1e-13) -> complex:
    """Refine a root estimate with a few Newton steps."""
    z = z0
    dp = np.asarray(np.polyder(p), dtype=complex)
    if dp.size == 0:
        return z

    for _ in range(max_iter):
        fz = horner_eval(p, z)
        dfz = horner_eval(dp, z)
        if abs(dfz) <= 1e-15:
            break
        step = fz / dfz
        z = z - step
        if abs(step) <= tol * (1.0 + abs(z)):
            break
    return z


def find_one_root_jenkins_traub_style(
    coeffs: Sequence[complex],
    tol: float = 1e-12,
    fixed_shift_steps: int = 5,
    max_variable_iter: int = 50,
    max_restarts: int = 16,
) -> complex:
    """Find one root using a Jenkins-Traub-style three-stage process."""
    p = trim_leading_zeros(coeffs)
    degree = p.size - 1
    if degree < 1:
        raise ValueError("Polynomial degree must be >= 1")
    if degree == 1:
        return -p[1] / p[0]

    p = p / p[0]  # normalize leading coefficient
    p_norm = float(np.linalg.norm(p, ord=1))
    radius = cauchy_root_bound(p)

    k0 = np.asarray(np.polyder(p), dtype=complex)
    if k0.size == 0:
        k0 = np.array([1.0 + 0.0j], dtype=complex)

    angles = np.linspace(0.0, 2.0 * np.pi, num=max_restarts, endpoint=False)

    for angle in angles:
        s = radius * np.exp(1j * (angle + 0.17))
        k = k0.copy()

        # Stage 2: fixed-shift updates
        for _ in range(fixed_shift_steps):
            k = jt_k_step(p, k, s)

        # Stage 3: variable-shift updates
        for _ in range(max_variable_iter):
            p_s = horner_eval(p, s)
            if abs(p_s) <= tol * (1.0 + p_norm):
                return s

            k_s = horner_eval(k, s)
            if abs(k_s) <= 1e-14:
                k = np.asarray(np.polyder(p), dtype=complex)
                k_s = horner_eval(k, s)
                if abs(k_s) <= 1e-14:
                    s = s + (0.01 + 0.02j)
                    continue

            delta = p_s / k_s
            s_next = s - delta
            k = jt_k_step(p, k, s_next)

            if abs(delta) <= tol * (1.0 + abs(s_next)):
                if abs(horner_eval(p, s_next)) <= 10.0 * tol * (1.0 + p_norm):
                    return s_next
            s = s_next

    # Fallback: damped Newton on original polynomial
    d = np.asarray(np.polyder(p), dtype=complex)
    s = radius * np.exp(1j * 0.31)
    for _ in range(max_variable_iter * 4):
        p_s = horner_eval(p, s)
        if abs(p_s) <= tol * (1.0 + p_norm):
            return s
        d_s = horner_eval(d, s)
        if abs(d_s) <= 1e-14:
            s = s + (0.05 + 0.05j)
            continue
        s = s - p_s / d_s

    raise RuntimeError("Failed to find a root in Jenkins-Traub-style iterations")


def jenkins_traub_roots(coeffs: Sequence[complex], tol: float = 1e-11) -> np.ndarray:
    """Compute all roots by repeated one-root extraction + deflation."""
    p = trim_leading_zeros(coeffs)
    if p.size <= 1:
        return np.array([], dtype=complex)

    roots: List[complex] = []

    while p.size > 3:
        root = find_one_root_jenkins_traub_style(p, tol=tol)
        root = newton_polish(p, root, tol=tol * 0.1)

        q, rem = synthetic_division(p, root)
        if abs(rem) > 1e3 * tol * (1.0 + np.linalg.norm(p, ord=1)):
            root = newton_polish(p, root, max_iter=40, tol=tol * 0.01)
            q, rem = synthetic_division(p, root)

        roots.append(root)
        p = trim_leading_zeros(q, tol=tol * 0.1)

    degree = p.size - 1
    if degree == 2:
        a, b, c = p
        disc = np.sqrt(b * b - 4.0 * a * c)
        roots.append((-b + disc) / (2.0 * a))
        roots.append((-b - disc) / (2.0 * a))
    elif degree == 1:
        roots.append(-p[1] / p[0])

    return np.asarray(roots, dtype=complex)


def evaluate_residuals(coeffs: Sequence[complex], roots: Iterable[complex]) -> np.ndarray:
    """Return |P(r_i)| for each computed root."""
    p = np.asarray(coeffs, dtype=complex)
    return np.asarray([abs(horner_eval(p, r)) for r in roots], dtype=float)


def match_roots_error(estimated: Sequence[complex], reference: Sequence[complex]) -> float:
    """Greedy matching max error between two root sets."""
    est = [complex(z) for z in estimated]
    ref = [complex(z) for z in reference]
    if len(est) != len(ref):
        return float("inf")

    max_err = 0.0
    remaining = ref.copy()
    for z in est:
        idx = min(range(len(remaining)), key=lambda i: abs(z - remaining[i]))
        err = abs(z - remaining[idx])
        max_err = max(max_err, float(err))
        remaining.pop(idx)
    return max_err


def fmt_complex(z: complex) -> str:
    """Compact complex formatter for console output."""
    return f"{z.real:+.10f}{z.imag:+.10f}j"


def run_case(name: str, coeffs: Sequence[complex]) -> None:
    """Run one deterministic test case and print diagnostics."""
    coeffs_arr = np.asarray(coeffs, dtype=complex)
    roots_jt = jenkins_traub_roots(coeffs_arr)
    roots_ref = np.roots(coeffs_arr)

    residuals = evaluate_residuals(coeffs_arr, roots_jt)
    max_residual = float(np.max(residuals)) if residuals.size else 0.0
    max_match_err = match_roots_error(roots_jt, roots_ref)

    print(f"[Case] {name}")
    print(f"  degree        : {len(coeffs_arr) - 1}")
    print(f"  max residual  : {max_residual:.3e}")
    print(f"  max match err : {max_match_err:.3e}")

    roots_sorted = sorted(roots_jt, key=lambda z: (round(z.real, 8), round(z.imag, 8)))
    for i, r in enumerate(roots_sorted, start=1):
        print(f"  jt_root[{i}]   : {fmt_complex(r)}")

    # Loose but meaningful MVP validation thresholds.
    assert max_residual < 1e-6, f"Residual too large in case '{name}': {max_residual}"
    assert max_match_err < 1e-5, f"Root mismatch too large in case '{name}': {max_match_err}"


def main() -> None:
    """Run non-interactive Jenkins-Traub MVP demo."""
    print("Jenkins-Traub MVP Demo (MATH-0127)")
    print("=" * 72)

    poly_from_roots = np.poly(np.array([-2.0, 0.5, 1.0 + 2.0j, 1.0 - 2.0j], dtype=complex))

    cases = [
        ("cubic_distinct_real", np.array([1.0, -6.0, 11.0, -6.0], dtype=complex)),
        ("cubic_with_complex_pair", np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)),
        ("quartic_mixed", poly_from_roots),
        ("quintic_unity", np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=complex)),
    ]

    for name, coeffs in cases:
        run_case(name, coeffs)
        print("-" * 72)

    print("All Jenkins-Traub MVP checks passed.")


if __name__ == "__main__":
    main()
