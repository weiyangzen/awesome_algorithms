"""Exponential Integrator MVP (ETD Euler) for a stiff semi-linear ODE system.

Model (after finite-difference spatial discretization of a 1D Allen-Cahn equation):
    u'(t) = A u(t) + g(u(t))
where:
    A = eps * L + I  (stiff linear part),
    g(u) = -u^3      (nonlinear local reaction).

The Exponential Time Differencing Euler (ETD-Euler) update is:
    u_{n+1} = exp(hA) u_n + h * phi_1(hA) g(u_n),
with phi_1(z) = (exp(z) - 1) / z.

We compute exp(hA) and h*phi_1(hA) from one block-matrix exponential:
    exp([[hA, hI],
         [ 0,  0]]) = [[exp(hA), h*phi_1(hA)],
                       [   0   ,     I      ]]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


@dataclass
class ExperimentResult:
    n: int
    t_end: float
    dt: float
    etd_rel_error: float
    explicit_rel_error: float
    etd_is_finite: bool
    explicit_is_finite: bool


def build_linear_operator(n: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct A = eps * L + I and return (A, grid_interior_points)."""
    if n < 3:
        raise ValueError("n must be >= 3")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    x = np.linspace(0.0, 1.0, n + 2)[1:-1]  # interior grid points only
    dx = 1.0 / (n + 1)

    main = -2.0 * np.ones(n, dtype=float)
    off = np.ones(n - 1, dtype=float)
    lap = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dx * dx)

    a = eps * lap + np.eye(n, dtype=float)
    return a, x


def nonlinear_term(u: np.ndarray) -> np.ndarray:
    """Nonlinear reaction term g(u) = -u^3 (component-wise)."""
    return -(u**3)


def rhs(_t: float, u: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Full RHS for reference solver."""
    return a @ u + nonlinear_term(u)


def precompute_etd_matrices(a: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute E=exp(dt*A) and H=h*phi_1(dt*A) from one block exponential."""
    n = a.shape[0]
    block = np.zeros((2 * n, 2 * n), dtype=float)
    block[:n, :n] = dt * a
    block[:n, n:] = dt * np.eye(n, dtype=float)
    em = expm(block)
    e = em[:n, :n]
    hphi1 = em[:n, n:]
    return e, hphi1


def etd_euler(a: np.ndarray, u0: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """ETD-Euler integrator for u' = A u + g(u)."""
    e, hphi1 = precompute_etd_matrices(a, dt)
    u = u0.astype(float).copy()
    for _ in range(steps):
        u = e @ u + hphi1 @ nonlinear_term(u)
    return u


def explicit_euler(a: np.ndarray, u0: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """Baseline explicit Euler; included to show stiffness impact."""
    u = u0.astype(float).copy()
    for _ in range(steps):
        u = u + dt * (a @ u + nonlinear_term(u))
    return u


def reference_solution(a: np.ndarray, u0: np.ndarray, t_end: float) -> np.ndarray:
    """High-accuracy reference via implicit Radau."""
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, a),
        t_span=(0.0, t_end),
        y0=u0,
        method="Radau",
        rtol=1e-9,
        atol=1e-11,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Reference solver failed: {sol.message}")
    return sol.y[:, -1]


def relative_l2_error(u: np.ndarray, ref: np.ndarray) -> float:
    den = float(np.linalg.norm(ref))
    if den == 0.0:
        return float(np.linalg.norm(u - ref))
    return float(np.linalg.norm(u - ref) / den)


def convergence_check(
    a: np.ndarray,
    u0: np.ndarray,
    t_end: float,
    u_ref: np.ndarray,
) -> list[tuple[float, float]]:
    """Return (dt, relative error vs a high-accuracy reference solution)."""
    dts = [0.1, 0.05, 0.025, 0.0125]
    rows: list[tuple[float, float]] = []
    for dt in dts:
        steps = int(round(t_end / dt))
        u = etd_euler(a, u0, dt, steps)
        rows.append((dt, relative_l2_error(u, u_ref)))
    return rows


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # Stiff semi-linear test case.
    n = 60
    eps = 0.01
    t_end = 1.0
    dt = 0.02
    steps = int(round(t_end / dt))

    a, x = build_linear_operator(n=n, eps=eps)
    u0 = 0.6 * np.sin(np.pi * x) + 0.2 * np.sin(3.0 * np.pi * x)

    u_ref = reference_solution(a, u0, t_end=t_end)
    u_etd = etd_euler(a, u0, dt=dt, steps=steps)
    u_explicit = explicit_euler(a, u0, dt=dt, steps=steps)

    result = ExperimentResult(
        n=n,
        t_end=t_end,
        dt=dt,
        etd_rel_error=relative_l2_error(u_etd, u_ref),
        explicit_rel_error=relative_l2_error(u_explicit, u_ref) if np.all(np.isfinite(u_explicit)) else float("nan"),
        etd_is_finite=bool(np.all(np.isfinite(u_etd))),
        explicit_is_finite=bool(np.all(np.isfinite(u_explicit))),
    )

    print("Exponential Integrator MVP: ETD Euler on a stiff semi-linear system")
    print("Model: u' = A u - u^3, where A = eps*L + I from 1D finite differences")
    print(
        f"Setup: n={result.n}, eps={eps:.4f}, T={result.t_end:.2f}, "
        f"dt={result.dt:.4f}, steps={steps}"
    )
    print()
    print("Reference: scipy.integrate.solve_ivp(method='Radau', rtol=1e-9, atol=1e-11)")
    print(f"ETD Euler finite: {result.etd_is_finite}, relative L2 error: {result.etd_rel_error:.6e}")
    print(
        "Explicit Euler finite: "
        f"{result.explicit_is_finite}, "
        f"relative L2 error: {result.explicit_rel_error:.6e}"
    )
    print()

    rows = convergence_check(a, u0, t_end=t_end, u_ref=u_ref)
    print("ETD Euler time-step convergence (vs Radau reference):")
    print("dt       rel_l2_error")
    print("---------------------------")
    for dt_i, err_i in rows:
        print(f"{dt_i:<8.4f} {err_i:>12.6e}")


if __name__ == "__main__":
    main()
