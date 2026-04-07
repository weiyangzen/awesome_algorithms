"""Minimal runnable MVP for wave function interpretation in 1D quantum well."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HBAR = 1.054_571_817e-34  # Reduced Planck constant (J*s).
M_E = 9.109_383_7015e-31  # Electron mass (kg).


@dataclass
class WaveFunctionResult:
    """Container for state, Born probability, and sampling diagnostics."""

    x: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    c_normalized: np.ndarray
    integral_rho: float
    x_mean: float
    x2_mean: float
    x_variance: float
    sampled_x: np.ndarray
    empirical_prob: np.ndarray
    theoretical_prob: np.ndarray
    l1_distance: float
    kl_divergence: float


def normalize_coefficients(c_raw: np.ndarray) -> np.ndarray:
    """Normalize complex amplitudes so that sum(|c_n|^2)=1."""
    c_raw = np.asarray(c_raw, dtype=np.complex128)
    if c_raw.ndim != 1 or c_raw.size == 0:
        raise ValueError("c_raw must be a non-empty 1D complex array.")
    if not np.all(np.isfinite(c_raw.real)) or not np.all(np.isfinite(c_raw.imag)):
        raise ValueError("c_raw contains non-finite values.")

    norm2 = float(np.sum(np.abs(c_raw) ** 2))
    if norm2 <= 0.0:
        raise ValueError("Coefficient norm must be positive.")
    return c_raw / np.sqrt(norm2)


def infinite_well_basis(n: int, x: np.ndarray, L: float) -> np.ndarray:
    """n-th stationary eigenfunction for 1D infinite square well on [0, L]."""
    if n < 1:
        raise ValueError("Quantum number n must be >= 1.")
    if L <= 0.0:
        raise ValueError("L must be positive.")
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


def build_wavefunction(
    x: np.ndarray,
    L: float,
    mass: float,
    c_normalized: np.ndarray,
    t: float,
    hbar: float,
) -> np.ndarray:
    """Construct psi(x,t)=sum_n c_n*phi_n(x)*exp(-iE_n t / hbar)."""
    if x.ndim != 1 or x.size < 16:
        raise ValueError("x must be a 1D grid with at least 16 points.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values.")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x grid must be strictly increasing.")
    if L <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("L, mass, and hbar must be positive.")

    psi = np.zeros_like(x, dtype=np.complex128)
    for idx, c_n in enumerate(c_normalized, start=1):
        n = idx
        E_n = (n**2) * (np.pi**2) * (hbar**2) / (2.0 * mass * (L**2))
        phase = np.exp(-1j * E_n * t / hbar)
        phi_n = infinite_well_basis(n=n, x=x, L=L)
        psi += c_n * phi_n * phase

    return psi


def probability_density(psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return normalized Born density rho(x)=|psi|^2."""
    if psi.ndim != 1 or x.ndim != 1 or psi.shape != x.shape:
        raise ValueError("psi and x must be same-shape 1D arrays.")

    rho = np.abs(psi) ** 2
    integral = float(np.trapezoid(rho, x))
    if integral <= 0.0 or not np.isfinite(integral):
        raise ValueError("Invalid rho integral; cannot normalize density.")
    rho /= integral
    return rho


def expectation_x_moments(x: np.ndarray, rho: np.ndarray) -> tuple[float, float, float]:
    """Compute <x>, <x^2>, and Var(x) from rho."""
    x_mean = float(np.trapezoid(x * rho, x))
    x2_mean = float(np.trapezoid((x**2) * rho, x))
    variance = float(max(x2_mean - x_mean**2, 0.0))
    return x_mean, x2_mean, variance


def discrete_distribution_from_density(x: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Convert continuous density into a discrete PMF on grid points."""
    # Midpoint-style bin widths associated with each grid point.
    dx = np.empty_like(x)
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]

    p = rho * dx
    p_sum = float(np.sum(p))
    if p_sum <= 0.0 or not np.isfinite(p_sum):
        raise ValueError("Invalid discrete probability mass.")
    p /= p_sum
    return p


def sample_position_measurements(
    x: np.ndarray,
    p_theoretical: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample position measurements according to Born PMF."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if x.ndim != 1 or p_theoretical.ndim != 1 or x.shape != p_theoretical.shape:
        raise ValueError("x and p_theoretical must be same-shape 1D arrays.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=n_samples, replace=True, p=p_theoretical)
    sampled_x = x[idx]
    empirical_prob = np.bincount(idx, minlength=x.size).astype(float) / float(n_samples)
    return sampled_x, empirical_prob


def compare_empirical_and_theoretical(
    empirical_prob: np.ndarray,
    theoretical_prob: np.ndarray,
) -> tuple[float, float]:
    """Compute L1 distance and KL divergence KL(empirical || theoretical)."""
    if empirical_prob.shape != theoretical_prob.shape:
        raise ValueError("Probability vectors must have same shape.")

    eps = 1e-15
    p = np.clip(empirical_prob, eps, 1.0)
    q = np.clip(theoretical_prob, eps, 1.0)
    p /= float(np.sum(p))
    q /= float(np.sum(q))

    l1 = float(np.sum(np.abs(p - q)))
    kl = float(np.sum(p * np.log(p / q)))
    return l1, kl


def run_wave_function_interpretation_mvp(
    L: float = 1.0e-9,
    mass: float = M_E,
    hbar: float = HBAR,
    t: float = 2.0e-16,
    n_grid: int = 512,
    n_samples: int = 30_000,
    seed: int = 23,
) -> WaveFunctionResult:
    """Execute full MVP: state construction, Born density, and measurement sampling."""
    if n_grid < 32:
        raise ValueError("n_grid must be >= 32.")
    if L <= 0.0:
        raise ValueError("L must be positive.")

    x = np.linspace(0.0, L, n_grid, dtype=float)

    c_raw = np.array(
        [
            1.0 + 0.0j,
            0.0 + 0.8j,
            0.6 - 0.3j,
        ],
        dtype=np.complex128,
    )
    c_norm = normalize_coefficients(c_raw)

    psi = build_wavefunction(x=x, L=L, mass=mass, c_normalized=c_norm, t=t, hbar=hbar)
    rho = probability_density(psi=psi, x=x)

    integral_rho = float(np.trapezoid(rho, x))
    x_mean, x2_mean, x_variance = expectation_x_moments(x=x, rho=rho)

    p_theoretical = discrete_distribution_from_density(x=x, rho=rho)
    sampled_x, empirical_prob = sample_position_measurements(
        x=x,
        p_theoretical=p_theoretical,
        n_samples=n_samples,
        seed=seed,
    )
    l1, kl = compare_empirical_and_theoretical(
        empirical_prob=empirical_prob,
        theoretical_prob=p_theoretical,
    )

    return WaveFunctionResult(
        x=x,
        psi=psi,
        rho=rho,
        c_normalized=c_norm,
        integral_rho=integral_rho,
        x_mean=x_mean,
        x2_mean=x2_mean,
        x_variance=x_variance,
        sampled_x=sampled_x,
        empirical_prob=empirical_prob,
        theoretical_prob=p_theoretical,
        l1_distance=l1,
        kl_divergence=kl,
    )


def run_checks(result: WaveFunctionResult, L: float) -> None:
    """Basic consistency checks for the MVP output."""
    if not np.isclose(result.integral_rho, 1.0, atol=2e-3):
        raise AssertionError(f"Normalization check failed: {result.integral_rho:.6f}")
    if not (0.0 <= result.x_mean <= L):
        raise AssertionError(f"<x> out of range: {result.x_mean:.6e}")
    if result.x_variance <= 0.0:
        raise AssertionError(f"Variance should be positive: {result.x_variance:.6e}")
    if result.l1_distance > 0.20:
        raise AssertionError(f"Empirical/theoretical L1 too large: {result.l1_distance:.6f}")
    if result.kl_divergence > 0.08:
        raise AssertionError(f"Empirical/theoretical KL too large: {result.kl_divergence:.6f}")


def preview_table(result: WaveFunctionResult, n_head: int = 5) -> pd.DataFrame:
    """Return top/bottom preview of the state and probability arrays."""
    df = pd.DataFrame(
        {
            "x_m": result.x,
            "psi_real": np.real(result.psi),
            "psi_imag": np.imag(result.psi),
            "rho_theoretical": result.rho,
            "p_empirical": result.empirical_prob,
        }
    )

    if 2 * n_head >= len(df):
        return df

    top = df.head(n_head)
    bottom = df.tail(n_head)
    ellipsis_row = pd.DataFrame([{col: np.nan for col in df.columns}])
    return pd.concat([top, ellipsis_row, bottom], ignore_index=True)


def main() -> None:
    L = 1.0e-9
    result = run_wave_function_interpretation_mvp(L=L)
    run_checks(result=result, L=L)

    std_x = float(np.sqrt(result.x_variance))

    print("Wave Function Interpretation MVP report")
    print(f"hbar (J*s)                      : {HBAR:.10e}")
    print(f"domain length L (m)             : {L:.3e}")
    print(f"basis coefficients c_n          : {result.c_normalized}")
    print(f"Integral rho dx                 : {result.integral_rho:.8f}")
    print(f"<x> (m)                         : {result.x_mean:.6e}")
    print(f"<x^2> (m^2)                     : {result.x2_mean:.6e}")
    print(f"std(x) (m)                      : {std_x:.6e}")
    print(f"L1(empirical, theoretical)      : {result.l1_distance:.6f}")
    print(f"KL(empirical || theoretical)    : {result.kl_divergence:.6f}")

    print("\nPreview (top/bottom rows):")
    preview = preview_table(result=result, n_head=4)
    print(preview.to_string(index=False, float_format=lambda v: f"{v: .3e}"))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
