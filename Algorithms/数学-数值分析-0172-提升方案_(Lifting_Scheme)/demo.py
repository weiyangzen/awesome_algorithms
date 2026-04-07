"""Minimal runnable MVP for Lifting Scheme (CDF 5/3) - MATH-0172."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass
class LiftingResult:
    """Packed multi-level lifting coefficients."""

    coeffs: np.ndarray
    levels: int
    coarse_size: int


def _validate_signal_and_levels(signal: np.ndarray, levels: int) -> None:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if signal.size == 0:
        raise ValueError("signal must be non-empty")

    n = int(signal.size)
    if n % 2 != 0:
        raise ValueError("signal length must be even")

    divisor = 2**levels
    if n % divisor != 0:
        raise ValueError(
            f"signal length={n} must be divisible by 2**levels={divisor}"
        )


def forward_lifting_cdf53_one_level(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-level CDF 5/3 forward lifting with edge replication."""
    if x.ndim != 1 or x.size % 2 != 0:
        raise ValueError("x must be 1D with even length")

    approx = x[0::2].astype(float, copy=True)
    detail = x[1::2].astype(float, copy=True)
    m = int(approx.size)

    # Predict step: detail <- odd - average(neighboring approx)
    for i in range(m):
        right = approx[i + 1] if i + 1 < m else approx[i]
        detail[i] = detail[i] - 0.5 * (approx[i] + right)

    # Update step: approx <- approx + quarter(neighboring detail sum)
    for i in range(m):
        left_detail = detail[i - 1] if i - 1 >= 0 else detail[0]
        approx[i] = approx[i] + 0.25 * (left_detail + detail[i])

    return approx, detail


def inverse_lifting_cdf53_one_level(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """One-level inverse CDF 5/3 lifting with edge replication."""
    if approx.ndim != 1 or detail.ndim != 1 or approx.size != detail.size:
        raise ValueError("approx and detail must be 1D arrays of equal length")

    s = approx.astype(float, copy=True)
    d = detail.astype(float, copy=True)
    m = int(s.size)

    # Undo update.
    for i in range(m):
        left_detail = d[i - 1] if i - 1 >= 0 else d[0]
        s[i] = s[i] - 0.25 * (left_detail + d[i])

    # Undo predict.
    for i in range(m):
        right = s[i + 1] if i + 1 < m else s[i]
        d[i] = d[i] + 0.5 * (s[i] + right)

    out = np.empty(2 * m, dtype=float)
    out[0::2] = s
    out[1::2] = d
    return out


def forward_lifting_cdf53(signal: np.ndarray, levels: int) -> LiftingResult:
    """Multi-level forward lifting, packed in-place layout [approx | detail]."""
    x = np.asarray(signal, dtype=float).copy()
    _validate_signal_and_levels(x, levels)

    current_n = int(x.size)
    for _ in range(levels):
        approx, detail = forward_lifting_cdf53_one_level(x[:current_n])
        half = current_n // 2
        x[:half] = approx
        x[half:current_n] = detail
        current_n = half

    return LiftingResult(coeffs=x, levels=levels, coarse_size=current_n)


def inverse_lifting_cdf53(coeffs: np.ndarray, levels: int) -> np.ndarray:
    """Inverse of packed multi-level CDF 5/3 lifting coefficients."""
    c = np.asarray(coeffs, dtype=float).copy()
    _validate_signal_and_levels(c, levels)

    current_n = int(c.size // (2**levels))
    for _ in range(levels):
        full = current_n * 2
        approx = c[:current_n]
        detail = c[current_n:full]
        reconstructed = inverse_lifting_cdf53_one_level(approx, detail)
        c[:full] = reconstructed
        current_n = full

    return c


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Soft-thresholding used by wavelet denoising."""
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def make_test_signal(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-smooth signal for denoising/compression demo."""
    if n <= 0:
        raise ValueError("n must be positive")

    t = np.linspace(0.0, 1.0, n, endpoint=False)
    signal = np.sin(2.0 * math.pi * 5.0 * t) + 0.55 * np.sin(2.0 * math.pi * 35.0 * t)
    signal = signal + 0.8 * (t > 0.35) - 0.5 * (t > 0.70)
    return t, signal


def run_reconstruction_demo() -> None:
    print("\n[Demo 1] Perfect reconstruction check")
    rng = np.random.default_rng(7)
    x = rng.normal(size=256)

    result = forward_lifting_cdf53(x, levels=4)
    x_hat = inverse_lifting_cdf53(result.coeffs, levels=result.levels)

    rel_err = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
    print(f"relative_reconstruction_error = {rel_err:.3e}")
    assert rel_err < 1e-11, f"reconstruction error too large: {rel_err}"


def run_denoising_demo() -> None:
    print("\n[Demo 2] Wavelet denoising with soft-thresholding")
    rng = np.random.default_rng(42)

    _, clean = make_test_signal(512)
    noisy = clean + 0.22 * rng.normal(size=clean.size)

    result = forward_lifting_cdf53(noisy, levels=4)
    coeffs = result.coeffs.copy()

    detail = coeffs[result.coarse_size :]
    sigma_hat = float(np.median(np.abs(detail)) / 0.6745 + 1e-12)
    # For this biorthogonal lifting setup, a moderated universal threshold
    # is more stable than the full universal threshold.
    tau = 0.2 * sigma_hat * math.sqrt(2.0 * math.log(detail.size))

    coeffs[result.coarse_size :] = soft_threshold(detail, tau)
    denoised = inverse_lifting_cdf53(coeffs, levels=result.levels)

    mse_noisy = float(np.mean((noisy - clean) ** 2))
    mse_denoised = float(np.mean((denoised - clean) ** 2))

    print(f"mse_noisy    = {mse_noisy:.6e}")
    print(f"mse_denoised = {mse_denoised:.6e}")
    print(f"threshold_tau= {tau:.6e}")

    assert mse_denoised < mse_noisy, "denoising did not improve MSE"


def run_sparse_compression_demo() -> None:
    print("\n[Demo 3] Sparse coefficient compression")

    _, signal = make_test_signal(512)
    result = forward_lifting_cdf53(signal, levels=4)
    coeffs = result.coeffs.copy()

    detail = coeffs[result.coarse_size :]
    keep_ratio = 0.25
    keep_count = max(1, int(keep_ratio * detail.size))

    idx = np.argpartition(np.abs(detail), -keep_count)[-keep_count:]
    mask = np.zeros(detail.size, dtype=bool)
    mask[idx] = True
    detail_sparse = np.where(mask, detail, 0.0)
    coeffs[result.coarse_size :] = detail_sparse

    reconstructed = inverse_lifting_cdf53(coeffs, levels=result.levels)
    rel_err = float(np.linalg.norm(reconstructed - signal) / np.linalg.norm(signal))
    sparsity = 1.0 - float(np.count_nonzero(coeffs) / coeffs.size)

    print(f"kept_detail_ratio = {keep_ratio:.2f}")
    print(f"global_sparsity   = {sparsity:.2%}")
    print(f"relative_error    = {rel_err:.6e}")

    assert sparsity > 0.4, "compression not sparse enough"


def main() -> None:
    print("Lifting Scheme MVP (CDF 5/3) - MATH-0172")
    print("=" * 72)

    run_reconstruction_demo()
    run_denoising_demo()
    run_sparse_compression_demo()

    print("=" * 72)
    print("All checks passed.")


if __name__ == "__main__":
    main()
