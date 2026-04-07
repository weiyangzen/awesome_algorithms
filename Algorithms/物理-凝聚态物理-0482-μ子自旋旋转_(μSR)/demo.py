"""Minimal runnable MVP for muon spin rotation (muSR) time-spectrum fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# muon gyromagnetic ratio: gamma_mu / (2*pi) in MHz/T.
GAMMA_MU_MHZ_PER_T = 135.5


@dataclass(frozen=True)
class MuSRParams:
    """Parameter bundle for synthetic muSR data generation."""

    a0: float = 0.24
    lambd: float = 0.45  # 1/us
    freq_mhz: float = 5.20  # precession frequency
    phi: float = 0.30  # rad
    a_bg: float = 0.015


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters and diagnostics."""

    a0: float
    lambd: float
    freq_mhz: float
    phi: float
    a_bg: float
    rmse: float
    r2: float



def mu_sr_asymmetry(t_us: np.ndarray, a0: float, lambd: float, freq_mhz: float, phi: float, a_bg: float) -> np.ndarray:
    """Transverse-field-like damped oscillation model.

    A(t) = A0 * exp(-lambda * t) * cos(2*pi*f*t + phi) + A_bg
    where t in microseconds and f in MHz.
    """
    return a0 * np.exp(-lambd * t_us) * np.cos(2.0 * np.pi * freq_mhz * t_us + phi) + a_bg



def simulate_data(params: MuSRParams, t_us: np.ndarray, noise_sigma: float, seed: int = 7) -> np.ndarray:
    """Create synthetic muSR asymmetry with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    clean = mu_sr_asymmetry(t_us, params.a0, params.lambd, params.freq_mhz, params.phi, params.a_bg)
    return clean + rng.normal(0.0, noise_sigma, size=t_us.shape)



def estimate_initial_guess(t_us: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate initial parameters using FFT peak + Hilbert-envelope decay."""
    dt = float(np.mean(np.diff(t_us)))
    if dt <= 0.0:
        raise ValueError("Time axis must be strictly increasing.")

    tail = max(8, len(y) // 10)
    a_bg0 = float(np.mean(y[-tail:]))
    y_centered = y - a_bg0

    # Frequency guess from real FFT peak.
    spectrum = np.fft.rfft(y_centered)
    freqs = np.fft.rfftfreq(len(y_centered), d=dt)  # MHz when t is in us
    if len(freqs) < 2:
        raise ValueError("Need at least two samples for frequency estimation.")
    peak_idx = 1 + int(np.argmax(np.abs(spectrum[1:])))
    freq0 = float(max(freqs[peak_idx], 0.05))

    # Amplitude guess from robust percentile spread.
    a0_0 = float(np.percentile(np.abs(y_centered), 90))
    a0_0 = max(a0_0, 1e-3)

    # Decay guess from Hilbert envelope log-linear regression.
    envelope = np.abs(signal.hilbert(y_centered))
    envelope = np.maximum(envelope, 1e-6)
    design = t_us.reshape(-1, 1)
    target = np.log(envelope)
    reg = LinearRegression().fit(design, target)
    lambd0 = float(max(-reg.coef_[0], 1e-3))

    # Phase guess from first sample relation.
    ratio = np.clip((y[0] - a_bg0) / a0_0, -1.0, 1.0)
    phi0 = float(np.arccos(ratio))

    return np.array([a0_0, lambd0, freq0, phi0, a_bg0], dtype=float)



def fit_mu_sr(t_us: np.ndarray, y: np.ndarray, init: np.ndarray | None = None) -> FitResult:
    """Fit muSR model with bounded nonlinear least squares."""
    if init is None:
        init = estimate_initial_guess(t_us, y)

    lower = np.array([0.0, 0.0, 0.01, -np.pi, -0.2], dtype=float)
    upper = np.array([1.0, 10.0, 20.0, np.pi, 0.2], dtype=float)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = mu_sr_asymmetry(t_us, theta[0], theta[1], theta[2], theta[3], theta[4])
        return pred - y

    result = optimize.least_squares(
        residual,
        x0=init,
        bounds=(lower, upper),
        method="trf",
        loss="linear",
        jac="2-point",
        max_nfev=4000,
    )
    if not result.success:
        raise RuntimeError(f"Fit failed: {result.message}")

    theta = result.x
    y_hat = mu_sr_asymmetry(t_us, theta[0], theta[1], theta[2], theta[3], theta[4])
    rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
    r2 = float(r2_score(y, y_hat))

    return FitResult(
        a0=float(theta[0]),
        lambd=float(theta[1]),
        freq_mhz=float(theta[2]),
        phi=float(theta[3]),
        a_bg=float(theta[4]),
        rmse=rmse,
        r2=r2,
    )



def summarize_fit(true_params: MuSRParams, fit: FitResult) -> pd.DataFrame:
    """Create a compact comparison table for true vs fitted parameters."""
    rows = [
        ("A0", true_params.a0, fit.a0, fit.a0 - true_params.a0),
        ("lambda (1/us)", true_params.lambd, fit.lambd, fit.lambd - true_params.lambd),
        ("f (MHz)", true_params.freq_mhz, fit.freq_mhz, fit.freq_mhz - true_params.freq_mhz),
        ("phi (rad)", true_params.phi, fit.phi, fit.phi - true_params.phi),
        ("A_bg", true_params.a_bg, fit.a_bg, fit.a_bg - true_params.a_bg),
    ]
    return pd.DataFrame(rows, columns=["parameter", "true", "fitted", "error"])



def run_demo() -> None:
    t_us = np.linspace(0.0, 8.0, 500)
    true_params = MuSRParams()
    noise_sigma = 0.012

    y = simulate_data(true_params, t_us, noise_sigma=noise_sigma, seed=2026)
    init = estimate_initial_guess(t_us, y)
    fit = fit_mu_sr(t_us, y, init=init)

    b_local_t = fit.freq_mhz / GAMMA_MU_MHZ_PER_T

    print("=== muSR MVP: Damped-precession fit ===")
    print(f"Samples: {len(t_us)}, time window: [{t_us[0]:.2f}, {t_us[-1]:.2f}] us")
    print(f"Noise sigma: {noise_sigma:.4f}")
    print("Initial guess [A0, lambda, f_MHz, phi, A_bg]:")
    print(np.array2string(init, precision=5, floatmode="fixed"))

    table = summarize_fit(true_params, fit)
    print("\nParameter recovery table:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\nFit diagnostics:")
    print(f"RMSE = {fit.rmse:.6f}")
    print(f"R^2  = {fit.r2:.6f}")
    print(f"Inferred local field B = f/gamma_mu = {b_local_t:.6f} T")



def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
