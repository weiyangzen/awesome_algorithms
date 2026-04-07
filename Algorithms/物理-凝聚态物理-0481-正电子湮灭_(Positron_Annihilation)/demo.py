"""Minimal runnable MVP for positron annihilation lifetime spectroscopy (PALS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from sklearn.metrics import mean_absolute_error, r2_score


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable softmax vector."""
    shifted = logits - np.max(logits)
    expv = np.exp(shifted)
    return expv / np.sum(expv)


def gaussian_irf(time_ns: np.ndarray, sigma_ns: float, center_ns: float) -> np.ndarray:
    """Construct a normalized Gaussian instrument response function."""
    if sigma_ns <= 0.0:
        raise ValueError("sigma_ns must be > 0")
    profile = np.exp(-0.5 * ((time_ns - center_ns) / sigma_ns) ** 2)
    norm = float(np.sum(profile))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("failed to normalize Gaussian IRF")
    return profile / norm


def intrinsic_annihilation_rate(
    time_ns: np.ndarray,
    lifetimes_ns: np.ndarray,
    intensities: np.ndarray,
) -> np.ndarray:
    """Mixture of exponential annihilation components (rate domain)."""
    if time_ns.ndim != 1:
        raise ValueError("time_ns must be a 1D array")
    if lifetimes_ns.ndim != 1 or intensities.ndim != 1:
        raise ValueError("lifetimes_ns and intensities must be 1D arrays")
    if lifetimes_ns.shape != intensities.shape:
        raise ValueError("lifetimes_ns and intensities must have the same length")
    if np.any(lifetimes_ns <= 0.0):
        raise ValueError("all lifetimes must be > 0")
    if np.any(intensities < 0.0):
        raise ValueError("intensities must be >= 0")

    weight_sum = float(np.sum(intensities))
    if not np.isclose(weight_sum, 1.0, atol=1e-9):
        raise ValueError("intensities must sum to 1")

    rates = np.zeros_like(time_ns, dtype=float)
    for tau, weight in zip(lifetimes_ns, intensities):
        rates += weight * np.exp(-time_ns / tau) / tau

    return rates


def convolved_shape(
    time_ns: np.ndarray,
    lifetimes_ns: np.ndarray,
    intensities: np.ndarray,
    sigma_ns: float,
    center_ns: float,
) -> np.ndarray:
    """Return normalized detector-smeared line shape over time bins."""
    if len(time_ns) < 2:
        raise ValueError("time_ns must contain at least two points")

    dt = float(time_ns[1] - time_ns[0])
    if dt <= 0.0:
        raise ValueError("time_ns must be strictly increasing")

    intrinsic = intrinsic_annihilation_rate(time_ns, lifetimes_ns, intensities)
    irf = gaussian_irf(time_ns, sigma_ns=sigma_ns, center_ns=center_ns)

    smeared = fftconvolve(intrinsic, irf, mode="full")[: len(time_ns)] * dt
    smeared = np.clip(smeared, 0.0, None)

    norm = float(np.sum(smeared))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("invalid convolved spectrum normalization")
    return smeared / norm


def expected_counts(
    time_ns: np.ndarray,
    lifetimes_ns: np.ndarray,
    intensities: np.ndarray,
    sigma_ns: float,
    center_ns: float,
    signal_counts: float,
    background_per_bin: float,
) -> np.ndarray:
    """Generate expected counts per bin from physical parameters."""
    if signal_counts <= 0.0:
        raise ValueError("signal_counts must be > 0")
    if background_per_bin < 0.0:
        raise ValueError("background_per_bin must be >= 0")

    shape = convolved_shape(
        time_ns=time_ns,
        lifetimes_ns=lifetimes_ns,
        intensities=intensities,
        sigma_ns=sigma_ns,
        center_ns=center_ns,
    )
    return signal_counts * shape + background_per_bin


def simulate_spectrum(
    rng: np.random.Generator,
    time_ns: np.ndarray,
    lifetimes_ns: np.ndarray,
    intensities: np.ndarray,
    sigma_ns: float,
    center_ns: float,
    signal_counts: float,
    background_per_bin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic noisy PALS spectrum with Poisson counting noise."""
    expectation = expected_counts(
        time_ns=time_ns,
        lifetimes_ns=lifetimes_ns,
        intensities=intensities,
        sigma_ns=sigma_ns,
        center_ns=center_ns,
        signal_counts=signal_counts,
        background_per_bin=background_per_bin,
    )
    observed = rng.poisson(expectation).astype(float)
    return expectation, observed


def unpack_parameters(params: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Map unconstrained optimization variables to physical parameters."""
    if params.shape != (2 * n_components + 2,):
        raise ValueError("parameter vector has an invalid length")

    lifetime_logs = params[:n_components]
    intensity_logits = params[n_components : 2 * n_components]
    signal_log = params[-2]
    background_log = params[-1]

    lifetimes_ns = np.exp(lifetime_logs)
    intensities = stable_softmax(intensity_logits)
    signal_counts = float(np.exp(signal_log))
    background_per_bin = float(np.exp(background_log))
    return lifetimes_ns, intensities, signal_counts, background_per_bin


def residual_vector(
    params: np.ndarray,
    time_ns: np.ndarray,
    observed_counts: np.ndarray,
    n_components: int,
    sigma_ns: float,
    center_ns: float,
) -> np.ndarray:
    """Weighted residual for nonlinear least squares fitting."""
    (
        lifetimes_ns,
        intensities,
        signal_counts,
        background_per_bin,
    ) = unpack_parameters(params, n_components)

    model_counts = expected_counts(
        time_ns=time_ns,
        lifetimes_ns=lifetimes_ns,
        intensities=intensities,
        sigma_ns=sigma_ns,
        center_ns=center_ns,
        signal_counts=signal_counts,
        background_per_bin=background_per_bin,
    )

    # Pearson-style weighting approximates Poisson variance stabilization.
    denom = np.sqrt(np.maximum(observed_counts, 1.0))
    return (model_counts - observed_counts) / denom


def fit_lifetime_spectrum(
    time_ns: np.ndarray,
    observed_counts: np.ndarray,
    n_components: int,
    sigma_ns: float,
    center_ns: float,
) -> dict[str, np.ndarray | float | bool | str]:
    """Fit lifetimes and intensities using constrained nonlinear least squares."""
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if time_ns.shape != observed_counts.shape:
        raise ValueError("time_ns and observed_counts must have the same shape")
    if np.any(observed_counts < 0.0):
        raise ValueError("observed_counts must be nonnegative")

    t_max = float(np.max(time_ns))

    init_lifetimes = np.linspace(0.08, max(0.4, 0.9 * t_max), n_components)
    init_logits = np.zeros(n_components)
    init_signal = max(float(np.sum(observed_counts)) * 0.9, 1.0)
    init_background = max(float(np.percentile(observed_counts, 5)), 0.2)

    x0 = np.concatenate(
        [
            np.log(init_lifetimes),
            init_logits,
            np.array([np.log(init_signal), np.log(init_background)], dtype=float),
        ]
    )

    lower = np.concatenate(
        [
            np.full(n_components, np.log(0.02)),
            np.full(n_components, -8.0),
            np.array([np.log(1.0), np.log(1e-6)], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.full(n_components, np.log(max(4.0 * t_max, 2.0))),
            np.full(n_components, 8.0),
            np.array(
                [
                    np.log(max(2.0 * np.sum(observed_counts), 10.0)),
                    np.log(max(np.max(observed_counts), 1.0)),
                ],
                dtype=float,
            ),
        ]
    )

    result = least_squares(
        fun=residual_vector,
        x0=x0,
        bounds=(lower, upper),
        args=(time_ns, observed_counts, n_components, sigma_ns, center_ns),
        method="trf",
        max_nfev=500,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )

    (
        lifetimes_ns,
        intensities,
        signal_counts,
        background_per_bin,
    ) = unpack_parameters(result.x, n_components)

    fitted_counts = expected_counts(
        time_ns=time_ns,
        lifetimes_ns=lifetimes_ns,
        intensities=intensities,
        sigma_ns=sigma_ns,
        center_ns=center_ns,
        signal_counts=signal_counts,
        background_per_bin=background_per_bin,
    )

    return {
        "lifetimes_ns": lifetimes_ns,
        "intensities": intensities,
        "signal_counts": signal_counts,
        "background_per_bin": background_per_bin,
        "fitted_counts": fitted_counts,
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
    }


def main() -> None:
    rng = np.random.default_rng(20260407)

    n_bins = 450
    dt_ns = 0.02
    time_ns = np.arange(n_bins, dtype=float) * dt_ns

    true_lifetimes_ns = np.array([0.125, 0.420, 1.950], dtype=float)
    true_intensities = np.array([0.58, 0.30, 0.12], dtype=float)
    sigma_ns = 0.080
    center_ns = 0.180
    true_signal_counts = 180_000.0
    true_background_per_bin = 3.0

    expected, observed = simulate_spectrum(
        rng=rng,
        time_ns=time_ns,
        lifetimes_ns=true_lifetimes_ns,
        intensities=true_intensities,
        sigma_ns=sigma_ns,
        center_ns=center_ns,
        signal_counts=true_signal_counts,
        background_per_bin=true_background_per_bin,
    )

    fit = fit_lifetime_spectrum(
        time_ns=time_ns,
        observed_counts=observed,
        n_components=3,
        sigma_ns=sigma_ns,
        center_ns=center_ns,
    )

    fit_lifetimes = np.asarray(fit["lifetimes_ns"], dtype=float)
    fit_intensities = np.asarray(fit["intensities"], dtype=float)
    fitted_counts = np.asarray(fit["fitted_counts"], dtype=float)

    order = np.argsort(fit_lifetimes)
    fit_lifetimes = fit_lifetimes[order]
    fit_intensities = fit_intensities[order]

    table = pd.DataFrame(
        {
            "component": np.arange(1, 4),
            "tau_true_ns": true_lifetimes_ns,
            "tau_fit_ns": fit_lifetimes,
            "tau_rel_err_%": 100.0 * (fit_lifetimes - true_lifetimes_ns) / true_lifetimes_ns,
            "I_true": true_intensities,
            "I_fit": fit_intensities,
            "I_abs_err": np.abs(fit_intensities - true_intensities),
        }
    )

    mae = float(mean_absolute_error(observed, fitted_counts))
    r2 = float(r2_score(observed, fitted_counts))
    dof = max(len(observed) - (2 * 3 + 2), 1)
    red_chi2 = float(np.sum((observed - fitted_counts) ** 2 / np.maximum(fitted_counts, 1.0)) / dof)

    print("Positron Annihilation MVP (PALS multi-exponential fit)")
    print("Model: [sum_k I_k * exp(-t/tau_k)/tau_k] (*) Gaussian IRF + flat background")
    print(f"bins={n_bins}, dt={dt_ns:.3f} ns, sigma={sigma_ns:.3f} ns, irf_center={center_ns:.3f} ns")
    print()
    print(f"optimizer_success = {fit['success']}")
    print(f"optimizer_message = {fit['message']}")
    print(f"fitted_signal_counts = {float(fit['signal_counts']):.1f} (true {true_signal_counts:.1f})")
    print(
        "fitted_background_per_bin = "
        f"{float(fit['background_per_bin']):.3f} (true {true_background_per_bin:.3f})"
    )
    print(f"fit_cost = {float(fit['cost']):.3f}")
    print(f"MAE(counts/bin) = {mae:.3f}")
    print(f"R2(observed vs fitted) = {r2:.6f}")
    print(f"reduced_chi2 = {red_chi2:.4f}")
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:10.5f}"))
    print()
    print("first_10_bins (observed, expected_true, fitted):")
    for i in range(10):
        print(
            f"bin {i:03d}: obs={observed[i]:8.2f}, "
            f"true={expected[i]:8.2f}, fit={fitted_counts[i]:8.2f}"
        )


if __name__ == "__main__":
    main()
