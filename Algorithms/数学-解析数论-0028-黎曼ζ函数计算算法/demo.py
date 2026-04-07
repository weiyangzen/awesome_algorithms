"""Riemann zeta function MVP.

Algorithmic core:
- Use the globally convergent Hasse series for the Dirichlet eta function.
- Recover zeta by zeta(s) = eta(s) / (1 - 2^(1-s)), s != 1.
- Search nontrivial zeros on the critical line Re(s)=1/2 by minimizing |zeta(1/2+it)|
  on coarse intervals and refining with golden-section search.

Run:
    python3 demo.py
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass, field
from math import comb, log, pi, sqrt
from typing import Dict, List, Tuple


def golden_section_minimize(
    func, a: float, b: float, iterations: int = 45
) -> Tuple[float, float]:
    """Minimize a unimodal scalar function on [a, b]."""
    gr = (sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c)
    fd = func(d)

    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = func(d)

    t_star = 0.5 * (a + b)
    return t_star, func(t_star)


@dataclass
class RiemannZetaHasse:
    """Compute zeta(s) via Hasse series and locate critical-line zeros."""

    n_terms: int = 90
    logs: List[float] = field(init=False)
    binom_rows: List[List[int]] = field(init=False)
    zeta_cache: Dict[Tuple[float, float], complex] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.n_terms < 8:
            raise ValueError("n_terms must be >= 8 for a meaningful MVP")

        self.logs = [log(k + 1.0) for k in range(self.n_terms)]
        self.binom_rows = [
            [comb(n, k) for k in range(n + 1)] for n in range(self.n_terms)
        ]

    @staticmethod
    def _key(s: complex) -> Tuple[float, float]:
        # Rounded cache key avoids duplicated evaluations from floating-point noise.
        return (round(s.real, 12), round(s.imag, 12))

    def eta(self, s: complex) -> complex:
        """Dirichlet eta(s) from the Hasse series."""
        powers = [cmath.exp(-s * self.logs[k]) for k in range(self.n_terms)]

        total = 0j
        half_pow = 0.5
        for n in range(self.n_terms):
            row = self.binom_rows[n]
            inner = 0j
            for k, coeff in enumerate(row):
                term = coeff * powers[k]
                if k & 1:
                    inner -= term
                else:
                    inner += term
            total += half_pow * inner
            half_pow *= 0.5

        return total

    def zeta(self, s: complex) -> complex:
        """Riemann zeta(s), excluding the pole at s=1."""
        if abs(s - 1.0) < 1e-12:
            raise ValueError("zeta(s) has a pole at s=1")

        key = self._key(s)
        cached = self.zeta_cache.get(key)
        if cached is not None:
            return cached

        eta_val = self.eta(s)
        denom = 1.0 - cmath.exp((1.0 - s) * log(2.0))
        if abs(denom) < 1e-15:
            raise ValueError("denominator too close to 0; choose another s")

        value = eta_val / denom
        self.zeta_cache[key] = value
        return value

    def critical_abs(self, t: float) -> float:
        """|zeta(1/2 + i t)| on the critical line."""
        return abs(self.zeta(0.5 + 1j * t))

    def find_critical_line_zeros(
        self,
        t_min: float = 10.0,
        t_max: float = 35.0,
        coarse_step: float = 0.25,
        max_zeros: int = 4,
        coarse_threshold: float = 0.25,
        refine_threshold: float = 0.03,
    ) -> List[Tuple[float, float]]:
        """Find approximate nontrivial zeros with t in [t_min, t_max].

        Returns:
            list of (t_star, |zeta(1/2+i*t_star)|)
        """
        if t_min >= t_max:
            raise ValueError("t_min must be less than t_max")
        if coarse_step <= 0:
            raise ValueError("coarse_step must be positive")

        # Coarse scan
        ts: List[float] = []
        t = t_min
        while t <= t_max + 1e-12:
            ts.append(round(t, 12))
            t += coarse_step

        vals = [self.critical_abs(x) for x in ts]

        # Local minima candidates
        candidates: List[Tuple[float, float]] = []
        for i in range(1, len(ts) - 1):
            if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1] and vals[i] < coarse_threshold:
                a, b = ts[i - 1], ts[i + 1]
                t_star, v_star = golden_section_minimize(self.critical_abs, a, b)
                if v_star < refine_threshold:
                    candidates.append((t_star, v_star))

        candidates.sort(key=lambda x: x[0])

        # Deduplicate close minima and keep strongest one
        deduped: List[Tuple[float, float]] = []
        for t_star, v_star in candidates:
            if deduped and abs(t_star - deduped[-1][0]) < 0.4:
                if v_star < deduped[-1][1]:
                    deduped[-1] = (t_star, v_star)
            else:
                deduped.append((t_star, v_star))

            if len(deduped) >= max_zeros:
                break

        return deduped


def main() -> None:
    solver = RiemannZetaHasse(n_terms=90)

    # Real-axis reference values.
    real_cases = [
        (2.0, pi**2 / 6.0, 1e-12),
        (4.0, pi**4 / 90.0, 1e-12),
        (0.0, -0.5, 1e-12),
        (-1.0, -1.0 / 12.0, 1e-10),
        (0.5, -1.4603545088095868, 1e-10),
    ]

    print("实轴数值校验:")
    all_ok = True
    for s_real, expected, tol in real_cases:
        got = solver.zeta(complex(s_real, 0.0)).real
        err = abs(got - expected)
        ok = err < tol
        all_ok = all_ok and ok
        tag = "OK" if ok else "FAIL"
        print(
            f"zeta({s_real:g}) = {got:.15f}, expected = {expected:.15f}, "
            f"err = {err:.3e} [{tag}]"
        )

    # Zero search on critical line.
    known_zeros = [
        14.134725141734693,
        21.022039638771554,
        25.01085758014569,
        30.424876125859513,
    ]
    found = solver.find_critical_line_zeros(
        t_min=10.0,
        t_max=35.0,
        coarse_step=0.25,
        max_zeros=4,
    )

    print("\n临界线非平凡零点近似 (Re(s)=1/2):")
    if not found:
        all_ok = False
        print("未找到候选零点，请增大 n_terms 或细化扫描步长。")
    else:
        for i, (t_star, abs_val) in enumerate(found, start=1):
            ref = known_zeros[i - 1] if i - 1 < len(known_zeros) else None
            if ref is None:
                print(f"zero #{i}: t ≈ {t_star:.12f}, |zeta| ≈ {abs_val:.3e}")
                continue
            dt = abs(t_star - ref)
            ok = dt < 5e-4 and abs_val < 1e-6
            all_ok = all_ok and ok
            tag = "OK" if ok else "WARN"
            print(
                f"zero #{i}: t ≈ {t_star:.12f}, reference = {ref:.12f}, "
                f"|Δt| = {dt:.3e}, |zeta| ≈ {abs_val:.3e} [{tag}]"
            )

    print("\n总体结论:", "通过" if all_ok else "存在偏差")


if __name__ == "__main__":
    main()
