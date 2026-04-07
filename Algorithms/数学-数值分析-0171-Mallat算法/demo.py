"""Mallat algorithm (Haar DWT) minimal runnable MVP.

This script demonstrates:
1) Multilevel Mallat decomposition,
2) Detail-coefficient soft-threshold denoising,
3) Inverse reconstruction and metric reporting.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional pretty-print dependency
    pd = None


def build_test_signal(
    n: int = 1024,
    seed: int = 42,
    noise_std: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic non-stationary signal and its noisy observation."""
    if n <= 0:
        raise ValueError("n must be positive")

    t = np.linspace(0.0, 1.0, n, endpoint=False)

    clean = (
        1.2 * np.sin(2.0 * np.pi * 5.0 * t)
        + 0.7 * np.sin(2.0 * np.pi * 31.0 * t)
        + 0.35 * np.cos(2.0 * np.pi * 67.0 * t)
    )

    step = np.where(t >= 0.55, 1.1, 0.0)
    pulse_1 = np.where((t >= 0.20) & (t < 0.215), 1.6, 0.0)
    pulse_2 = np.where((t >= 0.78) & (t < 0.79), -1.2, 0.0)
    clean = clean + step + pulse_1 + pulse_2

    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, noise_std, size=n)
    return t, clean, noisy


def mallat_haar_decompose(signal: np.ndarray, levels: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Perform multilevel Mallat decomposition with Haar wavelet.

    Returns:
    - Final approximation coefficients a_L
    - Detail coefficient list [d_1, d_2, ..., d_L]
    """
    x = np.asarray(signal, dtype=float)
    n = x.size

    if n == 0:
        raise ValueError("signal must be non-empty")
    if n & (n - 1):
        raise ValueError("signal length must be a power of 2 for this MVP")

    max_levels = int(math.log2(n))
    if levels < 1 or levels > max_levels:
        raise ValueError(f"levels must be in [1, {max_levels}], got {levels}")

    approx = x.copy()
    details: List[np.ndarray] = []
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for _ in range(levels):
        low = (approx[0::2] + approx[1::2]) * inv_sqrt2
        high = (approx[0::2] - approx[1::2]) * inv_sqrt2
        details.append(high)
        approx = low

    return approx, details


def mallat_haar_reconstruct(approx: np.ndarray, details: List[np.ndarray]) -> np.ndarray:
    """Reconstruct signal from Haar Mallat coefficients."""
    cur = np.asarray(approx, dtype=float).copy()
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    for d in reversed(details):
        det = np.asarray(d, dtype=float)
        if det.size != cur.size:
            raise ValueError(
                f"shape mismatch in reconstruction: approx={cur.size}, detail={det.size}"
            )
        up = np.empty(det.size * 2, dtype=float)
        up[0::2] = (cur + det) * inv_sqrt2
        up[1::2] = (cur - det) * inv_sqrt2
        cur = up

    return cur


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-threshold shrinkage."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio in dB against reference."""
    ref = np.asarray(reference)
    err = ref - np.asarray(estimate)
    signal_power = float(np.mean(ref**2))
    noise_power = float(np.mean(err**2))
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


def format_table(rows: List[Dict[str, float]]) -> str:
    """Pretty print rows with pandas if available, else plain text."""
    if pd is not None:
        return pd.DataFrame(rows).to_string(index=False)

    if not rows:
        return "(empty)"

    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(f"{r[h]}") for r in rows)) for h in headers}
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    split_line = "-+-".join("-" * widths[h] for h in headers)
    body = [" | ".join(f"{r[h]}".ljust(widths[h]) for h in headers) for r in rows]
    return "\n".join([header_line, split_line, *body])


def main() -> None:
    n = 1024
    levels = 5

    _, clean, noisy = build_test_signal(n=n, seed=42, noise_std=0.35)

    approx, details = mallat_haar_decompose(noisy, levels=levels)

    # Validate perfect reconstruction path before thresholding.
    reconstructed_noisy = mallat_haar_reconstruct(approx, details)
    recon_error = float(np.max(np.abs(reconstructed_noisy - noisy)))

    # MAD-based sigma estimate from the finest detail coefficients d1.
    sigma = float(np.median(np.abs(details[0])) / 0.6745)
    universal_threshold = float(sigma * math.sqrt(2.0 * math.log(n)))

    # Use scale-aware thresholds: deeper levels get slightly smaller thresholds.
    shrunk_details = [
        soft_threshold(d, universal_threshold / math.sqrt(level_idx))
        for level_idx, d in enumerate(details, start=1)
    ]

    denoised = mallat_haar_reconstruct(approx, shrunk_details)

    metrics = {
        "mse_noisy": mse(clean, noisy),
        "mse_denoised": mse(clean, denoised),
        "snr_noisy_db": snr_db(clean, noisy),
        "snr_denoised_db": snr_db(clean, denoised),
        "reconstruction_max_abs_error": recon_error,
        "sigma_est": sigma,
        "universal_threshold": universal_threshold,
    }
    metrics["snr_gain_db"] = metrics["snr_denoised_db"] - metrics["snr_noisy_db"]

    coeff_rows: List[Dict[str, float]] = []
    for i, d in enumerate(details, start=1):
        coeff_rows.append(
            {
                "level": i,
                "coeff_type": "detail",
                "length": int(d.size),
                "l2_norm": float(np.linalg.norm(d)),
            }
        )
    coeff_rows.append(
        {
            "level": levels,
            "coeff_type": "approx_final",
            "length": int(approx.size),
            "l2_norm": float(np.linalg.norm(approx)),
        }
    )

    sample_rows: List[Dict[str, float]] = []
    for i in range(10):
        sample_rows.append(
            {
                "idx": i,
                "clean": round(float(clean[i]), 6),
                "noisy": round(float(noisy[i]), 6),
                "denoised": round(float(denoised[i]), 6),
            }
        )

    print("Mallat (Haar DWT) MVP demo")
    print(f"signal_length={n}, levels={levels}")
    print()

    print("Coefficient summary:")
    print(format_table(coeff_rows))
    print()

    print("Metrics:")
    for key in [
        "mse_noisy",
        "mse_denoised",
        "snr_noisy_db",
        "snr_denoised_db",
        "snr_gain_db",
        "reconstruction_max_abs_error",
        "sigma_est",
        "universal_threshold",
    ]:
        print(f"- {key}: {metrics[key]:.6f}")
    print()

    print("First 10 samples (clean/noisy/denoised):")
    print(format_table(sample_rows))


if __name__ == "__main__":
    main()
