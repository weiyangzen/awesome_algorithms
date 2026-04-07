"""Minimal runnable MVP for the 1D Dirac equation.

The script builds a finite-difference Dirac Hamiltonian and runs three checks:
1) Spectrum and backend consistency (NumPy vs. PyTorch).
2) Plane-wave phase evolution against the discrete relativistic dispersion relation.
3) Gaussian packet evolution with norm/energy conservation and a zitterbewegung indicator.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm
from sklearn.linear_model import LinearRegression


def periodic_central_difference_matrix(num_points: int, dx: float) -> np.ndarray:
    """Return periodic first-derivative matrix using centered differences."""
    if num_points < 3:
        raise ValueError("num_points must be >= 3")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    d = np.zeros((num_points, num_points), dtype=np.float64)
    coeff = 0.5 / dx
    for j in range(num_points):
        d[j, (j + 1) % num_points] = coeff
        d[j, (j - 1) % num_points] = -coeff
    return d


def dirac_hamiltonian_1d(
    num_points: int,
    dx: float,
    mass: float,
    potential: np.ndarray | None = None,
) -> np.ndarray:
    """Build 1D Dirac Hamiltonian in natural units (hbar=c=1).

    H = -i * alpha * d/dx + beta * m + V(x) * I_2
    with alpha = sigma_x, beta = sigma_z.
    """
    if mass <= 0.0:
        raise ValueError("mass must be positive")

    d = periodic_central_difference_matrix(num_points=num_points, dx=dx)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    i_n = np.eye(num_points, dtype=np.complex128)

    if potential is None:
        v = np.zeros(num_points, dtype=np.float64)
    else:
        v = np.asarray(potential, dtype=np.float64)
        if v.ndim != 1 or v.shape[0] != num_points:
            raise ValueError("potential must be a 1-D array with length num_points")

    kinetic = -1j * np.kron(sigma_x, d)
    mass_term = float(mass) * np.kron(sigma_z, i_n)
    potential_term = np.kron(np.eye(2, dtype=np.complex128), np.diag(v))
    h = kinetic + mass_term + potential_term
    return h


def split_spin_components(state: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split flattened state [psi_up(x), psi_down(x)] into two components."""
    if state.ndim != 1 or state.shape[0] != 2 * num_points:
        raise ValueError("state must be 1-D with length 2 * num_points")
    return state[:num_points], state[num_points:]


def normalize_state(state: np.ndarray, dx: float) -> np.ndarray:
    """Normalize state with discrete integral sum |psi|^2 dx = 1."""
    norm = np.sqrt(float(np.sum(np.abs(state) ** 2) * dx))
    if norm == 0.0:
        raise ValueError("state norm is zero")
    return state / norm


def probability_density(state: np.ndarray, num_points: int) -> np.ndarray:
    """Return rho(x) = |psi_up|^2 + |psi_down|^2."""
    up, down = split_spin_components(state=state, num_points=num_points)
    return np.abs(up) ** 2 + np.abs(down) ** 2


def expectation_x(state: np.ndarray, x: np.ndarray, dx: float) -> float:
    """Return expectation value <x>."""
    rho = probability_density(state=state, num_points=x.shape[0])
    return float(np.sum(rho * x) * dx)


def expectation_energy(state: np.ndarray, hamiltonian: np.ndarray, dx: float) -> float:
    """Return real part of <psi|H|psi> under discrete integration."""
    value = np.vdot(state, hamiltonian @ state) * dx
    return float(np.real(value))


def build_plane_wave_state(x: np.ndarray, momentum: float, spinor: np.ndarray) -> np.ndarray:
    """Construct spinor plane wave psi(x) = u * exp(i p x)."""
    phase = np.exp(1j * float(momentum) * x)
    up = complex(spinor[0]) * phase
    down = complex(spinor[1]) * phase
    return np.concatenate([up, down])


def positive_energy_spinor(momentum: float, mass: float) -> np.ndarray:
    """Return normalized positive-energy spinor for 1D free Dirac Hamiltonian."""
    e = np.sqrt(momentum * momentum + mass * mass)
    raw = np.array([e + mass, momentum], dtype=np.complex128)
    return raw / np.linalg.norm(raw)


def gaussian_wave_packet(
    x: np.ndarray,
    x0: float,
    sigma: float,
    p0: float,
    spinor: np.ndarray,
) -> np.ndarray:
    """Return Gaussian spinor packet psi(x) = u * exp(-(x-x0)^2/(4 sigma^2)) * exp(i p0 x)."""
    envelope = np.exp(-((x - float(x0)) ** 2) / (4.0 * sigma * sigma))
    phase = np.exp(1j * float(p0) * x)
    scalar = envelope * phase
    up = complex(spinor[0]) * scalar
    down = complex(spinor[1]) * scalar
    return np.concatenate([up, down])


def run_spectrum_consistency_demo() -> Dict[str, float]:
    """Check Hermiticity, particle/antiparticle spectral symmetry, and NumPy/Torch consistency."""
    num_points = 72
    length = 24.0
    dx = length / num_points
    mass = 1.1
    h = dirac_hamiltonian_1d(num_points=num_points, dx=dx, mass=mass, potential=None)

    hermitian_defect = float(np.max(np.abs(h - h.conj().T)))
    evals_np = np.linalg.eigvalsh(h)

    h_torch = torch.tensor(h, dtype=torch.complex128)
    evals_torch = torch.linalg.eigvalsh(h_torch).detach().cpu().numpy()
    backend_gap = float(np.max(np.abs(evals_np - evals_torch)))

    pair_symmetry = float(np.max(np.abs(evals_np + evals_np[::-1])))

    assert hermitian_defect < 1e-12, f"Hermitian defect too large: {hermitian_defect:.3e}"
    assert backend_gap < 1e-10, f"NumPy/Torch spectrum mismatch: {backend_gap:.3e}"
    assert pair_symmetry < 1e-10, f"Particle-antiparticle symmetry broken: {pair_symmetry:.3e}"

    return {
        "grid_points": float(num_points),
        "dx": float(dx),
        "mass": float(mass),
        "hermitian_defect": hermitian_defect,
        "backend_gap": backend_gap,
        "pair_symmetry": pair_symmetry,
        "eval_min": float(np.min(evals_np)),
        "eval_max": float(np.max(evals_np)),
    }


def run_plane_wave_phase_demo() -> Dict[str, float]:
    """Propagate a plane-wave eigenstate and estimate energy from overlap phase."""
    num_points = 128
    length = 40.0
    dx = length / num_points
    x = np.linspace(-0.5 * length, 0.5 * length, num_points, endpoint=False, dtype=np.float64)

    mass = 0.9
    mode = 5
    k = 2.0 * np.pi * mode / length
    p_discrete = np.sin(k * dx) / dx
    e_ref = float(np.sqrt(mass * mass + p_discrete * p_discrete))

    spinor = positive_energy_spinor(momentum=p_discrete, mass=mass)
    state0 = build_plane_wave_state(x=x, momentum=k, spinor=spinor)
    state0 = normalize_state(state=state0, dx=dx)

    h = dirac_hamiltonian_1d(num_points=num_points, dx=dx, mass=mass, potential=None)
    dt = 0.02
    num_steps = 260
    u = expm(-1j * h * dt)

    times = np.arange(num_steps + 1, dtype=np.float64) * dt
    phase = np.zeros(num_steps + 1, dtype=np.float64)
    overlap_abs = np.zeros(num_steps + 1, dtype=np.float64)
    norms = np.zeros(num_steps + 1, dtype=np.float64)

    state = state0.copy()
    for i in range(num_steps + 1):
        overlap = np.vdot(state0, state) * dx
        phase[i] = float(np.angle(overlap))
        overlap_abs[i] = float(np.abs(overlap))
        norms[i] = float(np.sum(np.abs(state) ** 2) * dx)
        if i < num_steps:
            state = u @ state

    phase_unwrapped = np.unwrap(phase)
    model = LinearRegression()
    model.fit(times.reshape(-1, 1), phase_unwrapped)
    slope = float(model.coef_[0])
    e_est = -slope
    fit_r2 = float(model.score(times.reshape(-1, 1), phase_unwrapped))
    norm_drift = float(np.max(np.abs(norms - 1.0)))
    overlap_amp_drift = float(np.max(np.abs(overlap_abs - 1.0)))

    assert abs(e_est - e_ref) < 2e-3, f"Energy estimate mismatch: |{e_est:.6f}-{e_ref:.6f}| too large"
    assert norm_drift < 1e-11, f"Norm drift too large: {norm_drift:.3e}"
    assert overlap_amp_drift < 1e-11, f"Overlap amplitude drift too large: {overlap_amp_drift:.3e}"
    assert fit_r2 > 0.9999, f"Phase fit R^2 too low: {fit_r2:.6f}"

    return {
        "mode": float(mode),
        "mass": float(mass),
        "discrete_momentum": float(p_discrete),
        "energy_ref": e_ref,
        "energy_est": float(e_est),
        "energy_abs_err": float(abs(e_est - e_ref)),
        "phase_fit_r2": fit_r2,
        "norm_drift": norm_drift,
        "overlap_amp_drift": overlap_amp_drift,
    }


def run_gaussian_packet_demo() -> Dict[str, float]:
    """Propagate a Gaussian packet and report conservation + zitterbewegung indicator."""
    num_points = 192
    length = 60.0
    dx = length / num_points
    x = np.linspace(-0.5 * length, 0.5 * length, num_points, endpoint=False, dtype=np.float64)

    mass = 1.0
    h = dirac_hamiltonian_1d(num_points=num_points, dx=dx, mass=mass, potential=None)
    dt = 0.03
    num_steps = 220
    u = expm(-1j * h * dt)

    state = gaussian_wave_packet(
        x=x,
        x0=-8.0,
        sigma=1.8,
        p0=1.4,
        spinor=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
    )
    state = normalize_state(state=state, dx=dx)

    times = np.arange(num_steps + 1, dtype=np.float64) * dt
    x_mean = np.zeros(num_steps + 1, dtype=np.float64)
    norms = np.zeros(num_steps + 1, dtype=np.float64)
    energies = np.zeros(num_steps + 1, dtype=np.float64)

    for i in range(num_steps + 1):
        x_mean[i] = expectation_x(state=state, x=x, dx=dx)
        norms[i] = float(np.sum(np.abs(state) ** 2) * dx)
        energies[i] = expectation_energy(state=state, hamiltonian=h, dx=dx)
        if i < num_steps:
            state = u @ state

    x_model = LinearRegression()
    x_model.fit(times.reshape(-1, 1), x_mean)
    x_trend = x_model.predict(times.reshape(-1, 1))
    residual = x_mean - x_trend

    jitter_rms = float(np.sqrt(np.mean(residual * residual)))
    norm_drift = float(np.max(np.abs(norms - norms[0])))
    energy_drift = float(np.max(np.abs(energies - energies[0])))
    transport_slope = float(x_model.coef_[0])

    assert norm_drift < 1e-10, f"Packet norm drift too large: {norm_drift:.3e}"
    assert energy_drift < 1e-10, f"Energy drift too large: {energy_drift:.3e}"
    assert jitter_rms > 2e-3, f"Zitterbewegung indicator too weak: {jitter_rms:.3e}"

    return {
        "mass": float(mass),
        "dt": float(dt),
        "steps": float(num_steps),
        "x_start": float(x_mean[0]),
        "x_end": float(x_mean[-1]),
        "transport_slope": transport_slope,
        "zitterbewegung_rms": jitter_rms,
        "norm_drift": norm_drift,
        "energy_drift": energy_drift,
    }


def print_report(title: str, metrics: Dict[str, float]) -> None:
    """Pretty-print a one-row metrics table."""
    print(f"\n=== {title} ===")
    frame = pd.DataFrame([metrics])
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(frame.to_string(index=False))


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    torch.set_default_dtype(torch.float64)

    spectrum_metrics = run_spectrum_consistency_demo()
    phase_metrics = run_plane_wave_phase_demo()
    packet_metrics = run_gaussian_packet_demo()

    print_report("Demo A: Spectrum Consistency", spectrum_metrics)
    print_report("Demo B: Plane-Wave Phase Evolution", phase_metrics)
    print_report("Demo C: Gaussian Packet Dynamics", packet_metrics)

    print("\nAll Dirac equation MVP checks passed.")


if __name__ == "__main__":
    main()
