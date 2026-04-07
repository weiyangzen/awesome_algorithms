"""Minimal runnable MVP for the Heisenberg uncertainty principle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft, fftfreq, ifft


@dataclass(frozen=True)
class UncertaintyResult:
    """Container of uncertainty diagnostics for one quantum state."""

    name: str
    x_mean: float
    p_mean: float
    delta_x: float
    delta_p: float
    product: float
    bound: float


def normalize_wavefunction(psi: np.ndarray, dx: float) -> np.ndarray:
    """Normalize wavefunction so that integral |psi|^2 dx = 1."""
    norm = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("wavefunction norm must be positive and finite")
    return psi / norm


def gaussian_wave_packet(
    x: np.ndarray,
    x0: float,
    sigma: float,
    k0: float,
) -> np.ndarray:
    """Create a Gaussian wave packet exp(-(x-x0)^2/(2*sigma^2)) * exp(i*k0*x)."""
    envelope = np.exp(-((x - x0) ** 2) / (2.0 * sigma * sigma))
    phase = np.exp(1j * k0 * x)
    return envelope * phase


def bimodal_superposition(
    x: np.ndarray,
    sigma: float,
    separation: float,
    k0: float,
    relative_phase: float,
) -> np.ndarray:
    """Build a non-Gaussian state by superposing two displaced packets."""
    left = gaussian_wave_packet(x, x0=-separation / 2.0, sigma=sigma, k0=+k0)
    right = gaussian_wave_packet(x, x0=+separation / 2.0, sigma=sigma, k0=-k0)
    return left + np.exp(1j * relative_phase) * right


def spectral_derivatives(psi: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute first and second spatial derivatives via FFT spectral method."""
    n = psi.size
    k = 2.0 * np.pi * fftfreq(n, d=dx)
    psi_k = fft(psi)
    dpsi = ifft(1j * k * psi_k)
    d2psi = ifft(-(k**2) * psi_k)
    return dpsi, d2psi


def uncertainty_from_state(
    x: np.ndarray,
    psi: np.ndarray,
    hbar: float,
) -> UncertaintyResult:
    """Compute <x>, <p>, Delta x, Delta p, and product Delta x * Delta p."""
    if x.ndim != 1 or psi.ndim != 1 or x.size != psi.size:
        raise ValueError("x and psi must be 1D arrays with the same length")
    if x.size < 8:
        raise ValueError("grid must have at least 8 points")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive")

    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx):
        raise ValueError("x grid must be uniform")

    psi = normalize_wavefunction(psi, dx)
    dpsi, d2psi = spectral_derivatives(psi, dx)

    prob = np.abs(psi) ** 2
    x_mean = float(np.sum(x * prob) * dx)
    x2_mean = float(np.sum((x**2) * prob) * dx)
    var_x = max(x2_mean - x_mean * x_mean, 0.0)
    delta_x = float(np.sqrt(var_x))

    p_psi = -1j * hbar * dpsi
    p2_psi = -(hbar**2) * d2psi
    p_mean = float(np.real(np.vdot(psi, p_psi) * dx))
    p2_mean = float(np.real(np.vdot(psi, p2_psi) * dx))
    var_p = max(p2_mean - p_mean * p_mean, 0.0)
    delta_p = float(np.sqrt(var_p))

    bound = 0.5 * hbar
    product = delta_x * delta_p
    return UncertaintyResult(
        name="",
        x_mean=x_mean,
        p_mean=p_mean,
        delta_x=delta_x,
        delta_p=delta_p,
        product=product,
        bound=bound,
    )


def run_case(
    name: str,
    x: np.ndarray,
    psi: np.ndarray,
    hbar: float,
) -> UncertaintyResult:
    """Evaluate one named wavefunction case."""
    metrics = uncertainty_from_state(x=x, psi=psi, hbar=hbar)
    return UncertaintyResult(
        name=name,
        x_mean=metrics.x_mean,
        p_mean=metrics.p_mean,
        delta_x=metrics.delta_x,
        delta_p=metrics.delta_p,
        product=metrics.product,
        bound=metrics.bound,
    )


def print_report(results: list[UncertaintyResult]) -> None:
    """Print compact table for uncertainty diagnostics."""
    print("Heisenberg uncertainty principle demo")
    print("Check: Delta x * Delta p >= hbar / 2")
    print()
    print(
        "state                    <x>        <p>      Delta x    Delta p   "
        "Delta x*Delta p   (hbar/2)   ratio"
    )
    for item in results:
        ratio = item.product / item.bound
        print(
            f"{item.name:22s} "
            f"{item.x_mean:9.4f} {item.p_mean:10.4f} {item.delta_x:10.4f} "
            f"{item.delta_p:9.4f} {item.product:15.6f} {item.bound:10.6f} {ratio:8.4f}"
        )


def run_checks(results: list[UncertaintyResult]) -> None:
    """Run minimal numerical sanity checks."""
    if len(results) != 2:
        raise ValueError("expected exactly two benchmark cases")

    gaussian = results[0]
    non_gaussian = results[1]

    if gaussian.product < gaussian.bound * (1.0 - 1e-3):
        raise AssertionError("Gaussian packet appears to violate the uncertainty bound")

    if abs(gaussian.product / gaussian.bound - 1.0) > 0.08:
        raise AssertionError("Gaussian packet should stay close to the minimum uncertainty limit")

    if non_gaussian.product <= gaussian.product:
        raise AssertionError("Non-Gaussian superposition should have larger uncertainty product")


def main() -> None:
    hbar = 1.0
    n_grid = 4096
    x_min, x_max = -40.0, 40.0
    x = np.linspace(x_min, x_max, n_grid, endpoint=False)

    psi_gaussian = gaussian_wave_packet(x, x0=-6.0, sigma=1.1, k0=2.2)
    psi_bimodal = bimodal_superposition(
        x,
        sigma=0.9,
        separation=10.0,
        k0=1.5,
        relative_phase=0.6,
    )

    results = [
        run_case("gaussian packet", x, psi_gaussian, hbar),
        run_case("bimodal superposition", x, psi_bimodal, hbar),
    ]

    print_report(results)
    run_checks(results)
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
