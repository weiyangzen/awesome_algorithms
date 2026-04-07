"""Minimal runnable MVP for Decoherence (single-qubit pure dephasing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm
from sklearn.metrics import mean_absolute_error


def ket_to_density(ket: np.ndarray) -> np.ndarray:
    """Build density matrix rho = |psi><psi| from a state vector."""
    ket = np.asarray(ket, dtype=np.complex128).reshape(-1)
    if ket.shape != (2,):
        raise ValueError("ket must be a 2-dimensional complex vector for one qubit")
    norm = np.linalg.norm(ket)
    if norm <= 0:
        raise ValueError("ket norm must be positive")
    ket = ket / norm
    return np.outer(ket, ket.conjugate())


def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-9) -> bool:
    """Check Hermiticity, trace=1, and positive semidefiniteness."""
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (2, 2):
        return False
    if not np.allclose(rho, rho.conjugate().T, atol=tol):
        return False
    if not np.allclose(np.trace(rho), 1.0, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(rho)
    return bool(np.min(eigvals) >= -tol)


def pure_dephasing_channel(rho: np.ndarray, gamma: float, t: float) -> np.ndarray:
    """Apply pure-dephasing CPTP map: rho01(t)=rho01(0)*exp(-gamma t)."""
    if gamma < 0 or t < 0:
        raise ValueError("gamma and t must be non-negative")
    rho = np.asarray(rho, dtype=np.complex128)
    if not is_valid_density_matrix(rho):
        raise ValueError("input rho is not a valid one-qubit density matrix")

    lam = float(np.exp(-gamma * t))
    p = (1.0 - lam) / 2.0
    i2 = np.eye(2, dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    k0 = np.sqrt(1.0 - p) * i2
    k1 = np.sqrt(p) * sigma_z

    out = k0 @ rho @ k0.conjugate().T + k1 @ rho @ k1.conjugate().T
    out = 0.5 * (out + out.conjugate().T)
    out /= np.trace(out)
    return out


def lindblad_reference(rho: np.ndarray, gamma: float, t: float) -> np.ndarray:
    """Reference by exponentiating vectorized Lindblad generator."""
    if gamma < 0 or t < 0:
        raise ValueError("gamma and t must be non-negative")
    vec0 = np.asarray(rho, dtype=np.complex128).reshape(4)
    # Vector order: [rho00, rho01, rho10, rho11]
    generator = np.diag([0.0, -gamma, -gamma, 0.0]).astype(np.complex128)
    vec_t = expm(generator * t) @ vec0
    out = vec_t.reshape(2, 2)
    out = 0.5 * (out + out.conjugate().T)
    out /= np.trace(out)
    return out


def coherence_l1(rho: np.ndarray) -> float:
    """l1 coherence for one qubit: sum of off-diagonal magnitudes."""
    return float(np.abs(rho[0, 1]) + np.abs(rho[1, 0]))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy_bits(rho: np.ndarray, eps: float = 1e-12) -> float:
    eigvals = np.clip(np.linalg.eigvalsh(rho), eps, 1.0)
    return float(-np.sum(eigvals * np.log2(eigvals)))


def bloch_components(rho: np.ndarray) -> tuple[float, float, float]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    bx = float(np.real(np.trace(rho @ sigma_x)))
    by = float(np.real(np.trace(rho @ sigma_y)))
    bz = float(np.real(np.trace(rho @ sigma_z)))
    return bx, by, bz


def run_experiment(gamma: float, t_max: float, n_steps: int) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    if gamma < 0 or t_max < 0 or n_steps < 2:
        raise ValueError("invalid experiment parameters")

    ket_plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho0 = ket_to_density(ket_plus)

    t_grid = np.linspace(0.0, t_max, n_steps)
    rows: list[dict[str, float]] = []

    reference_diffs: list[float] = []
    coherence_curve: list[float] = []

    for t in t_grid:
        rho_t = pure_dephasing_channel(rho0, gamma=gamma, t=float(t))
        rho_ref = lindblad_reference(rho0, gamma=gamma, t=float(t))
        diff = float(np.linalg.norm(rho_t - rho_ref))
        reference_diffs.append(diff)

        c_l1 = coherence_l1(rho_t)
        coherence_curve.append(c_l1)

        bx, by, bz = bloch_components(rho_t)
        min_eig = float(np.min(np.linalg.eigvalsh(rho_t)))

        rows.append(
            {
                "t": float(t),
                "coherence_l1": c_l1,
                "purity": purity(rho_t),
                "entropy_bits": von_neumann_entropy_bits(rho_t),
                "bloch_x": bx,
                "bloch_y": by,
                "bloch_z": bz,
                "trace_real": float(np.real(np.trace(rho_t))),
                "min_eig": min_eig,
            }
        )

    df = pd.DataFrame(rows)

    expected_coherence = np.exp(-gamma * t_grid)
    mae_coherence = float(mean_absolute_error(expected_coherence, np.array(coherence_curve)))

    torch_expected = torch.exp(-gamma * torch.tensor(t_grid, dtype=torch.float64))
    torch_alignment_error = float(np.max(np.abs(torch_expected.numpy() - expected_coherence)))

    checks = {
        "max_reference_error": float(np.max(reference_diffs)),
        "coherence_mae": mae_coherence,
        "torch_alignment_error": torch_alignment_error,
        "trace_stays_one": bool(np.allclose(df["trace_real"].to_numpy(), 1.0, atol=1e-10)),
        "positive_semidefinite": bool(np.all(df["min_eig"].to_numpy() >= -1e-10)),
        "bloch_z_constant": bool(np.allclose(df["bloch_z"].to_numpy(), 0.0, atol=1e-10)),
        "all_checks_pass": False,
    }
    checks["all_checks_pass"] = bool(
        checks["max_reference_error"] < 1e-12
        and checks["coherence_mae"] < 1e-12
        and checks["torch_alignment_error"] < 1e-12
        and checks["trace_stays_one"]
        and checks["positive_semidefinite"]
        and checks["bloch_z_constant"]
    )

    return df, checks


def main() -> None:
    gamma = 0.8
    t_max = 6.0
    n_steps = 31

    df, checks = run_experiment(gamma=gamma, t_max=t_max, n_steps=n_steps)

    print("=== Decoherence MVP: Single-Qubit Pure Dephasing ===")
    print(f"gamma={gamma}, t_max={t_max}, n_steps={n_steps}")
    print()

    print("Trajectory sample (head):")
    print(df.head(6).to_string(index=False))
    print()

    print("Trajectory sample (tail):")
    print(df.tail(6).to_string(index=False))
    print()

    print("Checks:")
    for k, v in checks.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
