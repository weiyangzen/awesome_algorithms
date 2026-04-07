"""Dirichlet characters and L-function computation (runnable MVP).

Scope of this MVP:
- Build all Dirichlet characters modulo an odd prime p.
- Compute truncated Dirichlet L-series
      L_N(s, chi) = sum_{n=1..N} chi(n) / n^s,  s > 1.
- Verify two key properties numerically:
  1) Character orthogonality over (Z/pZ)^*.
  2) Principal-character identity
       L(s, chi_0) = zeta(s) * (1 - p^{-s})
     in truncated form:
       sum_{n<=N, p\nmid n} 1/n^s
       = sum_{n<=N} 1/n^s - p^{-s} * sum_{m<=N/p} 1/m^s.

Run:
    python3 demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import cmath
from typing import Dict, List

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


def is_prime(n: int) -> bool:
    """Deterministic primality check by trial division for small/medium n."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def factorize(n: int) -> Dict[int, int]:
    """Prime factorization via trial division: return {prime: exponent}."""
    if n <= 0:
        raise ValueError("factorize expects positive integer")

    x = n
    out: Dict[int, int] = {}

    while x % 2 == 0:
        out[2] = out.get(2, 0) + 1
        x //= 2

    f = 3
    while f * f <= x:
        while x % f == 0:
            out[f] = out.get(f, 0) + 1
            x //= f
        f += 2

    if x > 1:
        out[x] = out.get(x, 0) + 1

    return out


def primitive_root_prime(p: int) -> int:
    """Find one primitive root modulo odd prime p."""
    if not is_prime(p) or p == 2:
        raise ValueError("primitive_root_prime expects an odd prime")

    factors = sorted(factorize(p - 1).keys())
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    raise RuntimeError(f"No primitive root found for prime p={p}")


def format_complex(z: complex, digits: int = 8) -> str:
    """Compact formatter for complex numbers in result tables."""
    r = f"{z.real:.{digits}f}"
    i = f"{abs(z.imag):.{digits}f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{r} {sign} {i}i"


@dataclass
class DirichletCharactersPrime:
    """Dirichlet characters modulo an odd prime p.

    For p prime, (Z/pZ)^* is cyclic of size p-1.
    Let g be a primitive root and log_g(a) = m for a in [1, p-1].
    Then all characters are:
        chi_k(a) = exp(2*pi*i*k*m/(p-1)),  k=0..p-2
        chi_k(n) = 0 if p | n
    """

    p: int
    prefer_numpy: bool = True

    g: int = field(init=False)
    log_table: List[int] = field(init=False)
    _period_cache: Dict[int, List[complex]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not is_prime(self.p) or self.p == 2:
            raise ValueError("This MVP expects an odd prime modulus p")

        self.g = primitive_root_prime(self.p)
        self.log_table = self._build_discrete_log_table()

    def _build_discrete_log_table(self) -> List[int]:
        """Build table L where L[a] = m with g^m == a (mod p), for a in 1..p-1."""
        table = [-1] * self.p
        x = 1
        for m in range(self.p - 1):
            table[x] = m
            x = (x * self.g) % self.p

        if any(table[a] < 0 for a in range(1, self.p)):
            raise RuntimeError("Failed to build complete discrete-log table")
        return table

    def character_period(self, k: int) -> List[complex]:
        """Return one full period values [chi_k(0), chi_k(1), ..., chi_k(p-1)]."""
        kk = k % (self.p - 1)
        if kk in self._period_cache:
            return self._period_cache[kk]

        period = [0j] * self.p
        root = cmath.exp(2j * cmath.pi * kk / (self.p - 1))
        for a in range(1, self.p):
            period[a] = root ** self.log_table[a]

        self._period_cache[kk] = period
        return period

    def character_value(self, k: int, n: int) -> complex:
        """Evaluate chi_k(n)."""
        return self.character_period(k)[n % self.p]

    def l_series_truncated(self, k: int, s: float, n_terms: int) -> complex:
        """Compute truncated L_N(s, chi_k) for real s>1."""
        if s <= 1.0:
            raise ValueError("s must satisfy s > 1 for this series MVP")
        if n_terms < 1:
            raise ValueError("n_terms must be >= 1")

        period = self.character_period(k)

        if np is not None and self.prefer_numpy:
            idx = np.arange(1, n_terms + 1, dtype=np.int64)
            period_np = np.asarray(period, dtype=np.complex128)
            chi = period_np[idx % self.p]
            denom = np.power(idx.astype(np.float64), float(s))
            return complex(np.sum(chi / denom))

        total = 0j
        for n in range(1, n_terms + 1):
            total += period[n % self.p] / (n**s)
        return total

    def orthogonality_matrix(self) -> List[List[complex]]:
        """Matrix M[k][l] = sum_{a=1..p-1} chi_k(a) * conj(chi_l(a))."""
        m = self.p - 1
        periods = [self.character_period(k) for k in range(m)]

        matrix: List[List[complex]] = [[0j for _ in range(m)] for _ in range(m)]
        for k in range(m):
            for l in range(m):
                acc = 0j
                for a in range(1, self.p):
                    acc += periods[k][a] * periods[l][a].conjugate()
                matrix[k][l] = acc
        return matrix

    def principal_identity_residual(self, s: float, n_terms: int) -> complex:
        """Residual of truncated principal-character identity (should be ~0)."""
        left = self.l_series_truncated(k=0, s=s, n_terms=n_terms)

        if np is not None and self.prefer_numpy:
            n = np.arange(1, n_terms + 1, dtype=np.float64)
            zeta_n = float(np.sum(1.0 / np.power(n, s)))

            m_max = n_terms // self.p
            if m_max > 0:
                m = np.arange(1, m_max + 1, dtype=np.float64)
                zeta_m = float(np.sum(1.0 / np.power(m, s)))
            else:
                zeta_m = 0.0
        else:
            zeta_n = sum(1.0 / (n**s) for n in range(1, n_terms + 1))
            zeta_m = sum(1.0 / (m**s) for m in range(1, n_terms // self.p + 1))

        right = zeta_n - (self.p ** (-s)) * zeta_m
        return left - right


def demo_character_values(chars: DirichletCharactersPrime) -> None:
    """Print values chi_k(a) on reduced residues a=1..p-1."""
    p = chars.p
    print("=== Character Values On Residues ===")
    print(f"modulus p = {p}, primitive root g = {chars.g}")

    header = "a:" + "".join(f" {a:>16}" for a in range(1, p))
    print(header)
    print("-" * len(header))

    for k in range(p - 1):
        values = [chars.character_value(k, a) for a in range(1, p)]
        row = f"k={k}:" + "".join(f" {format_complex(v, digits=4):>16}" for v in values)
        print(row)


def demo_orthogonality(chars: DirichletCharactersPrime) -> None:
    """Show max diagonal/off-diagonal errors for orthogonality."""
    mat = chars.orthogonality_matrix()
    size = chars.p - 1

    max_diag_err = 0.0
    max_off_abs = 0.0
    for i in range(size):
        for j in range(size):
            if i == j:
                max_diag_err = max(max_diag_err, abs(mat[i][j] - size))
            else:
                max_off_abs = max(max_off_abs, abs(mat[i][j]))

    print("\n=== Orthogonality Check ===")
    print(f"expected diagonal value: {size}")
    print(f"max |diag - (p-1)| = {max_diag_err:.3e}")
    print(f"max |offdiag|       = {max_off_abs:.3e}")


def demo_l_values(chars: DirichletCharactersPrime, s_values: List[float], n_terms: int) -> None:
    """Compute and print truncated L-values for all characters."""
    print("\n=== Truncated L-Series Values ===")
    print(f"N = {n_terms}, modulus p = {chars.p}")

    for s in s_values:
        print(f"\ns = {s}")
        for k in range(chars.p - 1):
            val = chars.l_series_truncated(k=k, s=s, n_terms=n_terms)
            print(f"L_N({s}, chi_{k}) = {format_complex(val, digits=10)}")


def demo_principal_identity(chars: DirichletCharactersPrime, s: float, n_terms: int) -> None:
    """Check principal-character identity in truncated form."""
    residual = chars.principal_identity_residual(s=s, n_terms=n_terms)
    print("\n=== Principal Character Identity ===")
    print(f"check: L_N({s}, chi_0) - [sum(1/n^s) - p^(-s) sum(1/m^s)]")
    print(f"residual = {format_complex(residual, digits=12)}")
    print(f"|residual| = {abs(residual):.3e}")


def main() -> None:
    # Keep p and N modest so script remains fast even without NumPy.
    p = 7
    n_terms = 80_000
    s_values = [2.0, 3.0]

    chars = DirichletCharactersPrime(p=p, prefer_numpy=True)

    print("Dirichlet Characters and L-Function MVP")
    print(f"NumPy enabled: {np is not None and chars.prefer_numpy}")

    demo_character_values(chars)
    demo_orthogonality(chars)
    demo_l_values(chars, s_values=s_values, n_terms=n_terms)
    demo_principal_identity(chars, s=2.0, n_terms=n_terms)


if __name__ == "__main__":
    main()
