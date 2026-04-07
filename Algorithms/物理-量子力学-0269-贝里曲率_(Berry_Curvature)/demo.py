"""Berry curvature MVP on the Qi-Wu-Zhang Chern-insulator model.

This script computes lower-band Berry curvature in two independent ways:
1) Fukui-Hatsugai-Suzuki (FHS) gauge-invariant plaquette method
2) Closed-form two-band formula from d(k) · sigma

It then checks that the resulting Chern numbers match the known phase diagram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one Berry-curvature run."""

    mass: float
    nk: int = 61


def qiwuzhang_d_vector(kx: float, ky: float, mass: float) -> np.ndarray:
    """Return d(k) for H(k)=d_x sigma_x + d_y sigma_y + d_z sigma_z."""
    return np.array(
        [np.sin(kx), np.sin(ky), mass + np.cos(kx) + np.cos(ky)],
        dtype=np.float64,
    )


def qiwuzhang_hamiltonian(kx: float, ky: float, mass: float) -> np.ndarray:
    """2x2 Bloch Hamiltonian matrix at momentum (kx, ky)."""
    dx, dy, dz = qiwuzhang_d_vector(kx, ky, mass)
    return np.array(
        [[dz, dx - 1j * dy], [dx + 1j * dy, -dz]],
        dtype=np.complex128,
    )


def lower_band_state_and_gap(kx: float, ky: float, mass: float) -> tuple[np.ndarray, float]:
    """Return normalized lower-band eigenvector and band gap at one k-point."""
    eigenvalues, eigenvectors = np.linalg.eigh(qiwuzhang_hamiltonian(kx, ky, mass))
    state = eigenvectors[:, 0]  # np.linalg.eigh returns ascending eigenvalues.
    state = state / np.linalg.norm(state)
    gap = float(eigenvalues[1] - eigenvalues[0])
    return state, gap


def sample_lower_band_states(config: ModelConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample lower-band states on a periodic nk x nk Brillouin-zone grid."""
    ks = np.linspace(-np.pi, np.pi, config.nk, endpoint=False, dtype=np.float64)
    states = np.empty((config.nk, config.nk, 2), dtype=np.complex128)
    min_gap = np.inf
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            state, gap = lower_band_state_and_gap(float(kx), float(ky), config.mass)
            states[ix, iy] = state
            min_gap = min(min_gap, gap)
    return ks, states, float(min_gap)


def link_variable(state_a: np.ndarray, state_b: np.ndarray) -> complex:
    """Gauge-invariant U(1) link from two neighboring Bloch states."""
    overlap = np.vdot(state_a, state_b)
    norm = np.abs(overlap)
    if norm < 1e-14:
        raise ValueError("Encountered near-zero neighboring overlap; refine grid or avoid gap closing.")
    return overlap / norm


def berry_curvature_fhs(states: np.ndarray, dk: float) -> tuple[np.ndarray, float]:
    """Compute Berry curvature density and Chern number via FHS plaquette phases."""
    nk = states.shape[0]
    curvature = np.empty((nk, nk), dtype=np.float64)
    for ix in range(nk):
        ix_next = (ix + 1) % nk
        for iy in range(nk):
            iy_next = (iy + 1) % nk
            ux = link_variable(states[ix, iy], states[ix_next, iy])
            uy = link_variable(states[ix, iy], states[ix, iy_next])
            ux_y = link_variable(states[ix, iy_next], states[ix_next, iy_next])
            uy_x = link_variable(states[ix_next, iy], states[ix_next, iy_next])
            plaquette_phase = np.angle(ux * uy_x / (ux_y * uy))
            curvature[ix, iy] = plaquette_phase / (dk * dk)
    chern = float(np.sum(curvature) * dk * dk / (2.0 * np.pi))
    return curvature, chern


def analytic_berry_curvature(kx: float, ky: float, mass: float) -> float:
    """Closed-form lower-band Berry curvature for a two-level system."""
    d = qiwuzhang_d_vector(kx, ky, mass)
    d_kx = np.array([np.cos(kx), 0.0, -np.sin(kx)], dtype=np.float64)
    d_ky = np.array([0.0, np.cos(ky), -np.sin(ky)], dtype=np.float64)
    numerator = float(np.dot(d, np.cross(d_kx, d_ky)))
    denominator = float(np.linalg.norm(d) ** 3)
    # Sign convention is chosen to match the FHS plaquette orientation used above.
    return -0.5 * numerator / denominator


def berry_curvature_analytic(ks: np.ndarray, mass: float, dk: float) -> tuple[np.ndarray, float]:
    """Compute analytic Berry-curvature map and integrated Chern number."""
    nk = ks.shape[0]
    curvature = np.empty((nk, nk), dtype=np.float64)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            curvature[ix, iy] = analytic_berry_curvature(float(kx), float(ky), mass)
    chern = float(np.sum(curvature) * dk * dk / (2.0 * np.pi))
    return curvature, chern


def expected_chern_number(mass: float) -> int:
    """Known QWZ lower-band Chern number away from critical masses m in {-2,0,2}."""
    if -2.0 < mass < 0.0:
        return -1
    if 0.0 < mass < 2.0:
        return 1
    return 0


def run_single_case(config: ModelConfig) -> dict[str, float]:
    """Run one model configuration and return quantitative metrics."""
    ks, states, min_gap = sample_lower_band_states(config)
    dk = 2.0 * np.pi / config.nk
    fhs_curv, chern_fhs = berry_curvature_fhs(states, dk)
    ana_curv, chern_analytic = berry_curvature_analytic(ks, config.mass, dk)
    curvature_mae = float(np.mean(np.abs(fhs_curv - ana_curv)))
    return {
        "mass": config.mass,
        "chern_fhs": chern_fhs,
        "chern_analytic": chern_analytic,
        "chern_rounded": float(np.round(chern_fhs)),
        "min_gap": min_gap,
        "curvature_mae": curvature_mae,
    }


def main() -> None:
    phase_scan = [
        ModelConfig(mass=-3.0, nk=61),
        ModelConfig(mass=-1.0, nk=61),
        ModelConfig(mass=1.0, nk=61),
        ModelConfig(mass=3.0, nk=61),
    ]
    print("Berry curvature MVP on QWZ model")
    print("mass | Chern(FHS) | Chern(analytic) | rounded | min_gap | mean|Omega_fhs-Omega_an|")

    all_results = []
    for config in phase_scan:
        result = run_single_case(config)
        all_results.append(result)
        print(
            f"{result['mass']:>4.1f} | "
            f"{result['chern_fhs']:>10.6f} | "
            f"{result['chern_analytic']:>14.6f} | "
            f"{int(result['chern_rounded']):>7d} | "
            f"{result['min_gap']:>7.4f} | "
            f"{result['curvature_mae']:>10.6f}"
        )

    for result in all_results:
        expected = expected_chern_number(result["mass"])
        assert abs(result["chern_fhs"] - expected) < 0.05, "FHS Chern mismatch with phase diagram."
        assert abs(result["chern_analytic"] - expected) < 0.05, "Analytic Chern mismatch with phase diagram."
        assert abs(result["chern_fhs"] - result["chern_rounded"]) < 1e-8, "Chern should be quantized."
        assert result["min_gap"] > 0.1, "Configured mass is too close to a gap-closing transition."

    topological_case = next(r for r in all_results if np.isclose(r["mass"], 1.0))
    assert topological_case["curvature_mae"] < 0.08, "Discrete and analytic curvature maps disagree too much."
    print("All checks passed.")


if __name__ == "__main__":
    main()
