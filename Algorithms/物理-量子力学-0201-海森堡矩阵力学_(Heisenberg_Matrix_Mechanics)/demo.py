"""Heisenberg Matrix Mechanics MVP on a truncated harmonic oscillator Hilbert space.

This script demonstrates the matrix-mechanics viewpoint:
- Observables are matrices (operators) in a basis
- Dynamics uses the Heisenberg evolution O(t) = U† O U
- Canonical commutator and equations of motion are checked numerically

The demo is deterministic and needs no interactive input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm, norm
from sklearn.linear_model import LinearRegression


@dataclass
class HeisenbergResult:
    dim: int
    mass: float
    omega: float
    hbar: float
    commutator_error_numpy: float
    commutator_error_torch: float
    commutator_error_trunc_model: float
    energy_fit_slope: float
    energy_fit_intercept: float
    energy_fit_r2: float
    operator_error_table: pd.DataFrame
    expectation_table: pd.DataFrame


def validate_parameters(dim: int, mass: float, omega: float, hbar: float) -> None:
    if dim < 6:
        raise ValueError("dim must be >= 6 for a meaningful truncated matrix-mechanics demo.")
    if mass <= 0.0 or omega <= 0.0 or hbar <= 0.0:
        raise ValueError("mass, omega, hbar must all be positive.")


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def build_ladder_operators(dim: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(float(n))
    adag = a.conj().T
    return a, adag


def build_harmonic_operators(dim: int, mass: float, omega: float, hbar: float) -> dict[str, np.ndarray]:
    a, adag = build_ladder_operators(dim=dim)
    identity = np.eye(dim, dtype=np.complex128)
    number = adag @ a

    x_scale = np.sqrt(hbar / (2.0 * mass * omega))
    p_scale = 1j * np.sqrt(mass * hbar * omega / 2.0)

    x_op = x_scale * (a + adag)
    p_op = p_scale * (adag - a)
    h_op = hbar * omega * (number + 0.5 * identity)

    return {"I": identity, "a": a, "adag": adag, "X": x_op, "P": p_op, "H": h_op}


def heisenberg_evolve(operator: np.ndarray, h_op: np.ndarray, t: float, hbar: float) -> np.ndarray:
    u = expm((-1j / hbar) * h_op * t)
    return u.conj().T @ operator @ u


def harmonic_closed_form(
    x0: np.ndarray,
    p0: np.ndarray,
    mass: float,
    omega: float,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.cos(omega * t)
    s = np.sin(omega * t)
    x_t = c * x0 + (s / (mass * omega)) * p0
    p_t = c * p0 - (mass * omega * s) * x0
    return x_t, p_t


def coherent_state(alpha: complex, dim: int) -> np.ndarray:
    coeff = np.empty(dim, dtype=np.complex128)
    coeff[0] = np.exp(-0.5 * (abs(alpha) ** 2))
    for n in range(1, dim):
        coeff[n] = coeff[n - 1] * alpha / np.sqrt(float(n))
    coeff /= np.linalg.norm(coeff)
    return coeff


def expectation(psi: np.ndarray, operator: np.ndarray) -> complex:
    return np.vdot(psi, operator @ psi)


def build_operator_error_table(
    times: Sequence[float],
    x0: np.ndarray,
    p0: np.ndarray,
    h_op: np.ndarray,
    mass: float,
    omega: float,
    hbar: float,
    identity: np.ndarray,
    commutator_expected_trunc: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for t in times:
        x_num = heisenberg_evolve(operator=x0, h_op=h_op, t=float(t), hbar=hbar)
        p_num = heisenberg_evolve(operator=p0, h_op=h_op, t=float(t), hbar=hbar)

        x_exact, p_exact = harmonic_closed_form(x0=x0, p0=p0, mass=mass, omega=omega, t=float(t))
        x_err = float(norm(x_num - x_exact, ord="fro"))
        p_err = float(norm(p_num - p_exact, ord="fro"))

        comm_now = commutator(x_num, p_num)
        comm_err_ideal = float(norm(comm_now - 1j * hbar * identity, ord="fro"))
        comm_err_trunc = float(norm(comm_now - commutator_expected_trunc, ord="fro"))
        rows.append(
            {
                "t": float(t),
                "X_error_fro": x_err,
                "P_error_fro": p_err,
                "comm_error_to_ihI_fro": comm_err_ideal,
                "comm_error_to_trunc_model_fro": comm_err_trunc,
            }
        )

    return pd.DataFrame(rows)


def build_expectation_table(
    times: Sequence[float],
    psi0: np.ndarray,
    x0: np.ndarray,
    p0: np.ndarray,
    h_op: np.ndarray,
    mass: float,
    omega: float,
    hbar: float,
) -> pd.DataFrame:
    x_init = float(np.real_if_close(expectation(psi0, x0)))
    p_init = float(np.real_if_close(expectation(psi0, p0)))

    rows: list[dict[str, float]] = []
    for t in times:
        x_t = heisenberg_evolve(operator=x0, h_op=h_op, t=float(t), hbar=hbar)
        p_t = heisenberg_evolve(operator=p0, h_op=h_op, t=float(t), hbar=hbar)

        x_exp = float(np.real_if_close(expectation(psi0, x_t)))
        p_exp = float(np.real_if_close(expectation(psi0, p_t)))

        x_cls = x_init * np.cos(omega * t) + (p_init / (mass * omega)) * np.sin(omega * t)
        p_cls = p_init * np.cos(omega * t) - (mass * omega * x_init) * np.sin(omega * t)

        rows.append(
            {
                "t": float(t),
                "<X>(t)": x_exp,
                "X_classical": float(x_cls),
                "abs_X_diff": float(abs(x_exp - x_cls)),
                "<P>(t)": p_exp,
                "P_classical": float(p_cls),
                "abs_P_diff": float(abs(p_exp - p_cls)),
            }
        )

    return pd.DataFrame(rows)


def fit_energy_levels(h_op: np.ndarray) -> tuple[float, float, float]:
    energies = np.linalg.eigvalsh(h_op).real
    n_vals = np.arange(len(energies), dtype=float).reshape(-1, 1)

    model = LinearRegression()
    model.fit(n_vals, energies)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(n_vals, energies))
    return slope, intercept, r2


def torch_commutator_error(x_op: np.ndarray, p_op: np.ndarray, hbar: float) -> float:
    dim = x_op.shape[0]
    xt = torch.tensor(x_op, dtype=torch.complex128)
    pt = torch.tensor(p_op, dtype=torch.complex128)
    ident = torch.eye(dim, dtype=torch.complex128)
    comm = xt @ pt - pt @ xt
    return float(torch.linalg.norm(comm - 1j * hbar * ident).item())


def truncated_commutator_model(dim: int, hbar: float) -> np.ndarray:
    identity = np.eye(dim, dtype=np.complex128)
    edge_projector = np.zeros((dim, dim), dtype=np.complex128)
    edge_projector[-1, -1] = 1.0
    return 1j * hbar * (identity - dim * edge_projector)


def run_heisenberg_mvp() -> HeisenbergResult:
    dim = 24
    mass = 1.0
    omega = 1.7
    hbar = 1.0
    validate_parameters(dim=dim, mass=mass, omega=omega, hbar=hbar)

    ops = build_harmonic_operators(dim=dim, mass=mass, omega=omega, hbar=hbar)
    x_op = ops["X"]
    p_op = ops["P"]
    h_op = ops["H"]
    identity = ops["I"]

    comm_expected_trunc = truncated_commutator_model(dim=dim, hbar=hbar)
    comm_now = commutator(x_op, p_op)
    commutator_error_numpy = float(norm(comm_now - 1j * hbar * identity, ord="fro"))
    commutator_error_torch = torch_commutator_error(x_op=x_op, p_op=p_op, hbar=hbar)
    commutator_error_trunc_model = float(norm(comm_now - comm_expected_trunc, ord="fro"))

    slope, intercept, r2 = fit_energy_levels(h_op=h_op)

    period = 2.0 * np.pi / omega
    times = np.linspace(0.0, period, 9)
    op_table = build_operator_error_table(
        times=times,
        x0=x_op,
        p0=p_op,
        h_op=h_op,
        mass=mass,
        omega=omega,
        hbar=hbar,
        identity=identity,
        commutator_expected_trunc=comm_expected_trunc,
    )

    psi0 = coherent_state(alpha=1.1 + 0.2j, dim=dim)
    exp_table = build_expectation_table(
        times=times,
        psi0=psi0,
        x0=x_op,
        p0=p_op,
        h_op=h_op,
        mass=mass,
        omega=omega,
        hbar=hbar,
    )

    return HeisenbergResult(
        dim=dim,
        mass=mass,
        omega=omega,
        hbar=hbar,
        commutator_error_numpy=commutator_error_numpy,
        commutator_error_torch=commutator_error_torch,
        commutator_error_trunc_model=commutator_error_trunc_model,
        energy_fit_slope=slope,
        energy_fit_intercept=intercept,
        energy_fit_r2=r2,
        operator_error_table=op_table,
        expectation_table=exp_table,
    )


def main() -> None:
    result = run_heisenberg_mvp()

    print("Heisenberg Matrix Mechanics MVP (Harmonic Oscillator)")
    print(f"dim={result.dim}, mass={result.mass:.3f}, omega={result.omega:.3f}, hbar={result.hbar:.3f}")
    print(f"commutator_error_numpy = {result.commutator_error_numpy:.8e}")
    print(f"commutator_error_torch = {result.commutator_error_torch:.8e}")
    print(f"commutator_error_trunc_model = {result.commutator_error_trunc_model:.8e}")
    print(
        "energy_fit: "
        f"slope={result.energy_fit_slope:.8f} (expected hbar*omega={result.hbar * result.omega:.8f}), "
        f"intercept={result.energy_fit_intercept:.8f} (expected 0.5*hbar*omega={0.5 * result.hbar * result.omega:.8f}), "
        f"R2={result.energy_fit_r2:.8f}"
    )

    print("\nOperator-level errors over one period:")
    print(result.operator_error_table.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

    print("\nExpectation-level (Ehrenfest) check:")
    print(result.expectation_table.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

    all_finite = bool(np.isfinite(result.operator_error_table.to_numpy(dtype=float)).all())
    finite_exp = bool(np.isfinite(result.expectation_table.to_numpy(dtype=float)).all())
    print(f"\ncheck_operator_table_all_finite = {all_finite}")
    print(f"check_expectation_table_all_finite = {finite_exp}")


if __name__ == "__main__":
    main()
