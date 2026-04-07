"""Minimal runnable MVP for Symplectic Geometry.

This demo uses a 1D harmonic oscillator Hamiltonian system and compares:
1) explicit Euler (not symplectic)
2) symplectic Euler (semi-implicit, symplectic)

It validates geometric properties through the symplectic condition:
A^T J A = J
for each one-step Jacobian matrix A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IntegrationResult:
    method: str
    q: np.ndarray
    p: np.ndarray
    energy: np.ndarray
    max_energy_drift: float


def hamiltonian(q: np.ndarray, p: np.ndarray, omega: float) -> np.ndarray:
    """Return H(q, p) = 0.5 * (p^2 + omega^2 * q^2)."""
    return 0.5 * (p * p + (omega * q) * (omega * q))


def explicit_euler_step(q: float, p: float, h: float, omega: float) -> tuple[float, float]:
    """Forward Euler for Hamilton equations (non-symplectic)."""
    q_next = q + h * p
    p_next = p - h * (omega**2) * q
    return q_next, p_next


def symplectic_euler_step(q: float, p: float, h: float, omega: float) -> tuple[float, float]:
    """Kick-drift symplectic Euler update.

    p_{n+1} = p_n - h * dH/dq(q_n)
    q_{n+1} = q_n + h * dH/dp(p_{n+1})
    """
    p_next = p - h * (omega**2) * q
    q_next = q + h * p_next
    return q_next, p_next


def integrate(
    stepper,
    q0: float,
    p0: float,
    h: float,
    omega: float,
    n_steps: int,
    method: str,
) -> IntegrationResult:
    """Integrate for n_steps and record phase trajectory + energy."""
    q = np.empty(n_steps + 1, dtype=float)
    p = np.empty(n_steps + 1, dtype=float)
    q[0] = q0
    p[0] = p0

    for i in range(n_steps):
        q[i + 1], p[i + 1] = stepper(float(q[i]), float(p[i]), h, omega)

    energy = hamiltonian(q, p, omega)
    max_energy_drift = float(np.max(np.abs(energy - energy[0])))
    return IntegrationResult(
        method=method,
        q=q,
        p=p,
        energy=energy,
        max_energy_drift=max_energy_drift,
    )


def canonical_j() -> np.ndarray:
    """Canonical symplectic matrix J in 2D phase space."""
    return np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)


def jacobian_explicit_euler(h: float, omega: float) -> np.ndarray:
    """One-step Jacobian matrix of explicit Euler map."""
    return np.array([[1.0, h], [-h * (omega**2), 1.0]], dtype=float)


def jacobian_symplectic_euler(h: float, omega: float) -> np.ndarray:
    """One-step Jacobian matrix of kick-drift symplectic Euler map."""
    return np.array(
        [[1.0 - (h**2) * (omega**2), h], [-h * (omega**2), 1.0]],
        dtype=float,
    )


def symplecticity_error(a: np.ndarray) -> float:
    """Return ||A^T J A - J||_F."""
    j = canonical_j()
    return float(np.linalg.norm(a.T @ j @ a - j, ord="fro"))


def run_checks(
    explicit_result: IntegrationResult,
    symplectic_result: IntegrationResult,
    a_explicit: np.ndarray,
    a_symplectic: np.ndarray,
) -> dict[str, float]:
    """Validate geometric structure and expected qualitative behavior."""
    explicit_sym_err = symplecticity_error(a_explicit)
    symplectic_sym_err = symplecticity_error(a_symplectic)

    det_explicit = float(np.linalg.det(a_explicit))
    det_symplectic = float(np.linalg.det(a_symplectic))

    if symplectic_sym_err > 1e-12:
        raise AssertionError(
            f"Symplectic Euler should preserve symplectic form, got error={symplectic_sym_err:.3e}"
        )
    if explicit_sym_err < 1e-6:
        raise AssertionError(
            f"Explicit Euler should violate symplectic form, got error={explicit_sym_err:.3e}"
        )

    if abs(det_symplectic - 1.0) > 1e-12:
        raise AssertionError(
            f"Symplectic Euler determinant must be 1, got {det_symplectic:.12f}"
        )
    if det_explicit <= 1.0 + 1e-6:
        raise AssertionError(
            f"Explicit Euler determinant should be > 1 in this setup, got {det_explicit:.12f}"
        )

    if symplectic_result.max_energy_drift > 0.05:
        raise AssertionError(
            "Symplectic Euler energy drift unexpectedly large: "
            f"{symplectic_result.max_energy_drift:.3e}"
        )
    if explicit_result.max_energy_drift < 1.0:
        raise AssertionError(
            "Explicit Euler drift unexpectedly small: "
            f"{explicit_result.max_energy_drift:.3e}"
        )

    return {
        "explicit_symplecticity_error": explicit_sym_err,
        "symplectic_symplecticity_error": symplectic_sym_err,
        "det_explicit": det_explicit,
        "det_symplectic": det_symplectic,
    }


def main() -> None:
    omega = 1.0
    h = 0.1
    n_steps = 800
    q0 = 1.0
    p0 = 0.0

    explicit_result = integrate(
        stepper=explicit_euler_step,
        q0=q0,
        p0=p0,
        h=h,
        omega=omega,
        n_steps=n_steps,
        method="explicit_euler",
    )
    symplectic_result = integrate(
        stepper=symplectic_euler_step,
        q0=q0,
        p0=p0,
        h=h,
        omega=omega,
        n_steps=n_steps,
        method="symplectic_euler",
    )

    a_explicit = jacobian_explicit_euler(h=h, omega=omega)
    a_symplectic = jacobian_symplectic_euler(h=h, omega=omega)
    checks = run_checks(explicit_result, symplectic_result, a_explicit, a_symplectic)

    print("Symplectic Geometry MVP: harmonic oscillator")
    print(f"omega={omega}, step_size={h}, n_steps={n_steps}")
    print(f"initial_state=(q0={q0}, p0={p0})")
    print(f"explicit_max_energy_drift={explicit_result.max_energy_drift:.6e}")
    print(f"symplectic_max_energy_drift={symplectic_result.max_energy_drift:.6e}")
    print(f"explicit_final_state=(q={explicit_result.q[-1]:.6f}, p={explicit_result.p[-1]:.6f})")
    print(
        f"symplectic_final_state=(q={symplectic_result.q[-1]:.6f}, p={symplectic_result.p[-1]:.6f})"
    )
    print(f"explicit_symplecticity_error={checks['explicit_symplecticity_error']:.6e}")
    print(f"symplectic_symplecticity_error={checks['symplectic_symplecticity_error']:.6e}")
    print(f"det_explicit={checks['det_explicit']:.6f}")
    print(f"det_symplectic={checks['det_symplectic']:.6f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
