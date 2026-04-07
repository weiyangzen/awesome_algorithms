"""Minimal runnable MVP for the Interaction Picture.

Model:
- Two-level system with H = H0 + V, where
  H0 = (Delta / 2) * sigma_z, V = (Omega / 2) * sigma_x.
- Interaction picture dynamics:
  i d|psi_I>/dt = V_I(t)|psi_I>, V_I(t)=U0^dagger(t) V U0(t).
- Compare three trajectories:
  1) Interaction-picture ODE + back transform
  2) Exact Schrodinger evolution under H0 + V
  3) First-order Dyson approximation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def normalize_state(psi: np.ndarray) -> np.ndarray:
    """Return normalized state vector; raise on invalid input."""
    if psi.ndim != 1:
        raise ValueError("psi must be a 1D state vector")
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("state vector norm must be positive")
    return psi / norm


def validate_time_grid(t_eval: np.ndarray) -> None:
    """Validate the requested output time grid."""
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 points")
    if not np.all(np.diff(t_eval) >= 0.0):
        raise ValueError("t_eval must be non-decreasing")


def pack_state(psi: np.ndarray) -> np.ndarray:
    """Pack complex state into real vector [Re(psi), Im(psi)]."""
    return np.concatenate([psi.real, psi.imag])


def unpack_state(y: np.ndarray, n: int) -> np.ndarray:
    """Unpack real vector back into an n-dimensional complex state."""
    return y[:n] + 1j * y[n:]


def unitary_from_hamiltonian(h: np.ndarray, t: float, hbar: float = 1.0) -> np.ndarray:
    """Compute U(t)=exp(-i h t / hbar) for time-independent h."""
    return expm(-1j * h * t / hbar)


def interaction_hamiltonian(
    h0: np.ndarray,
    v: np.ndarray,
    t: float,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return V_I(t)=U0^dagger(t) V U0(t), where U0=exp(-iH0 t / hbar)."""
    u0 = unitary_from_hamiltonian(h0, t, hbar=hbar)
    return u0.conj().T @ v @ u0


def rhs_interaction_real(
    t: float,
    y: np.ndarray,
    h0: np.ndarray,
    v: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Real-valued RHS for i d|psi_I>/dt = V_I(t)|psi_I>."""
    n = h0.shape[0]
    psi_i = unpack_state(y, n)
    v_i = interaction_hamiltonian(h0, v, t, hbar=hbar)
    dpsi_i = -(1j / hbar) * (v_i @ psi_i)
    return pack_state(dpsi_i)


def integrate_interaction_state(
    psi0: np.ndarray,
    h0: np.ndarray,
    v: np.ndarray,
    t_eval: np.ndarray,
    hbar: float = 1.0,
) -> list[np.ndarray]:
    """Integrate interaction-picture state on t_eval."""
    validate_time_grid(t_eval)
    psi0 = normalize_state(psi0.astype(complex))

    solution = solve_ivp(
        fun=lambda t, y: rhs_interaction_real(t, y, h0, v, hbar),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=pack_state(psi0),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    n = psi0.size
    return [unpack_state(solution.y[:, k], n) for k in range(solution.y.shape[1])]


def recover_schrodinger_state(
    psi_i: np.ndarray,
    h0: np.ndarray,
    t: float,
    hbar: float = 1.0,
) -> np.ndarray:
    """Map interaction-picture state back: |psi_S> = U0 |psi_I>."""
    u0 = unitary_from_hamiltonian(h0, t, hbar=hbar)
    return u0 @ psi_i


def exact_schrodinger_state(
    psi0: np.ndarray,
    h_total: np.ndarray,
    t: float,
    hbar: float = 1.0,
) -> np.ndarray:
    """Exact state under time-independent full Hamiltonian H_total."""
    u_total = unitary_from_hamiltonian(h_total, t, hbar=hbar)
    return u_total @ psi0


def dyson_first_order_state(
    psi0: np.ndarray,
    h0: np.ndarray,
    v: np.ndarray,
    t: float,
    hbar: float = 1.0,
    n_steps: int = 400,
) -> np.ndarray:
    """First-order Dyson approximation in interaction picture.

    |psi_I(t)> ~= (I - i/hbar * integral_0^t V_I(t') dt') |psi(0)>
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    psi0 = normalize_state(psi0.astype(complex))
    if np.isclose(t, 0.0):
        return psi0.copy()

    ts = np.linspace(0.0, float(t), n_steps + 1)
    dt = ts[1] - ts[0]
    integral = np.zeros_like(h0, dtype=complex)

    for idx, tk in enumerate(ts):
        weight = 0.5 if idx in (0, n_steps) else 1.0
        integral += weight * interaction_hamiltonian(h0, v, float(tk), hbar=hbar)

    integral *= dt
    identity = np.eye(h0.shape[0], dtype=complex)
    return (identity - (1j / hbar) * integral) @ psi0


def excited_population(psi: np.ndarray) -> float:
    """Return population on basis state |1>."""
    return float(np.clip(np.abs(psi[1]) ** 2, 0.0, 1.0))


def norm_error(psi: np.ndarray) -> float:
    """Return |<psi|psi> - 1|."""
    return float(abs(np.vdot(psi, psi).real - 1.0))


def main() -> None:
    hbar = 1.0

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    delta = 2.0
    omega = 0.7

    h0 = 0.5 * delta * sigma_z
    v = 0.5 * omega * sigma_x
    h_total = h0 + v

    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    psi0 = normalize_state(psi0)

    t_eval = np.linspace(0.0, 8.0, 11)

    psi_i_traj = integrate_interaction_state(psi0, h0, v, t_eval, hbar=hbar)
    psi_from_i_traj = [
        recover_schrodinger_state(psi_i, h0, float(t), hbar=hbar)
        for t, psi_i in zip(t_eval, psi_i_traj)
    ]
    psi_exact_traj = [
        exact_schrodinger_state(psi0, h_total, float(t), hbar=hbar)
        for t in t_eval
    ]
    psi_dyson_traj = [
        recover_schrodinger_state(
            dyson_first_order_state(psi0, h0, v, float(t), hbar=hbar, n_steps=600),
            h0,
            float(t),
            hbar=hbar,
        )
        for t in t_eval
    ]

    rows: list[dict[str, float]] = []
    for t, psi_from_i, psi_exact, psi_dyson in zip(
        t_eval,
        psi_from_i_traj,
        psi_exact_traj,
        psi_dyson_traj,
    ):
        p1_exact = excited_population(psi_exact)
        p1_from_i = excited_population(psi_from_i)
        p1_dyson = excited_population(psi_dyson)
        rows.append(
            {
                "time": float(t),
                "P1_exact": p1_exact,
                "P1_from_I": p1_from_i,
                "abs_err": abs(p1_from_i - p1_exact),
                "P1_dyson1": p1_dyson,
                "dyson_abs_err": abs(p1_dyson - p1_exact),
                "state_l2_err": float(np.linalg.norm(psi_from_i - psi_exact)),
                "norm_err_I": norm_error(psi_from_i),
            }
        )

    table = pd.DataFrame(rows)

    print("Interaction Picture demo (two-level quantum system)")
    print("H0=(Delta/2)*sigma_z, V=(Omega/2)*sigma_x, H=H0+V")
    print(f"Delta={delta:.3f}, Omega={omega:.3f}, hbar={hbar:.3f}")
    print()
    print(
        table.to_string(
            index=False,
            formatters={
                "time": lambda x: f"{x:5.2f}",
                "P1_exact": lambda x: f"{x:10.6f}",
                "P1_from_I": lambda x: f"{x:10.6f}",
                "abs_err": lambda x: f"{x:10.3e}",
                "P1_dyson1": lambda x: f"{x:10.6f}",
                "dyson_abs_err": lambda x: f"{x:10.3e}",
                "state_l2_err": lambda x: f"{x:10.3e}",
                "norm_err_I": lambda x: f"{x:10.3e}",
            },
        )
    )

    print()
    print(f"max state_l2_err (interaction vs exact) = {table['state_l2_err'].max():.3e}")
    print(f"max population abs_err              = {table['abs_err'].max():.3e}")
    print(f"max Dyson first-order abs_err       = {table['dyson_abs_err'].max():.3e}")


if __name__ == "__main__":
    main()
