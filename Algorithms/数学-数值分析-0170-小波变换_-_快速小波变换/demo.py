"""Minimal runnable MVP: 1D Haar Fast Wavelet Transform (FWT)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)


@dataclass
class FWTResult:
    """Container for one forward+inverse run."""

    original_length: int
    padded_length: int
    levels: int
    approx: np.ndarray
    details: list[np.ndarray]  # details[0] is level-1 (finest)
    reconstructed: np.ndarray


def validate_signal(signal: np.ndarray) -> np.ndarray:
    """Ensure input is a finite 1D float array."""
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"signal must be 1D, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("signal must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal contains non-finite values")
    return arr


def next_power_of_two(n: int) -> int:
    """Return smallest power of two >= n for n>=1."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def pad_to_power_of_two(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero-pad signal to the next power-of-two length."""
    original_length = int(signal.size)
    target = next_power_of_two(original_length)
    if target == original_length:
        return signal.copy(), original_length
    padded = np.zeros(target, dtype=float)
    padded[:original_length] = signal
    return padded, original_length


def haar_fwt(signal: np.ndarray, levels: int | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute multilevel 1D Haar FWT. Return (final_approx, details_finest_to_coarsest)."""
    current = validate_signal(signal).copy()
    n = int(current.size)

    if n & (n - 1):
        raise ValueError("haar_fwt expects power-of-two length; call pad_to_power_of_two first")

    max_levels = int(math.log2(n))
    if levels is None:
        levels = max_levels
    if levels < 1 or levels > max_levels:
        raise ValueError(f"levels must be in [1, {max_levels}], got {levels}")

    details: list[np.ndarray] = []

    for _ in range(levels):
        even = current[0::2]
        odd = current[1::2]
        approx = (even + odd) / SQRT2
        detail = (even - odd) / SQRT2
        details.append(detail)
        current = approx

    return current, details


def haar_ifwt(approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    """Inverse multilevel 1D Haar transform from approx + details_finest_to_coarsest."""
    current = validate_signal(np.asarray(approx, dtype=float))
    if current.size == 0:
        raise ValueError("approx must be non-empty")

    # Inverse order: start from coarsest detail back to finest detail.
    for detail in reversed(details):
        d = validate_signal(np.asarray(detail, dtype=float))
        if d.size != current.size:
            raise ValueError(
                "detail/approx size mismatch during inverse transform: "
                f"detail={d.size}, approx={current.size}"
            )

        out = np.empty(current.size * 2, dtype=float)
        out[0::2] = (current + d) / SQRT2
        out[1::2] = (current - d) / SQRT2
        current = out

    return current


def soft_threshold(coeffs: np.ndarray, tau: float) -> np.ndarray:
    """Soft-threshold operator sign(x)*max(|x|-tau,0)."""
    c = np.asarray(coeffs, dtype=float)
    return np.sign(c) * np.maximum(np.abs(c) - tau, 0.0)


def universal_threshold(detail_finest: np.ndarray, n: int, scale: float = 1.0) -> float:
    """Estimate sigma by MAD and compute universal threshold."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    d = np.asarray(detail_finest, dtype=float)
    mad = float(np.median(np.abs(d)))
    sigma = mad / 0.6745 if mad > 0 else 0.0
    tau = scale * sigma * math.sqrt(2.0 * math.log(n))
    return float(tau)


def wavelet_denoise(
    noisy_signal: np.ndarray,
    levels: int,
    threshold_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Denoise by Haar FWT + soft-thresholding details + inverse."""
    validated = validate_signal(noisy_signal)
    padded, original_length = pad_to_power_of_two(validated)

    approx, details = haar_fwt(padded, levels=levels)
    tau = universal_threshold(details[0], n=original_length, scale=threshold_scale)
    denoised_details = [soft_threshold(d, tau) for d in details]

    reconstructed = haar_ifwt(approx, denoised_details)
    return reconstructed[:original_length], tau


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def build_demo_signal(n: int = 150, seed: int = 2026, noise_std: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic clean/noisy pair for reproducible demo."""
    if n < 16:
        raise ValueError("n must be >= 16 for this demo")

    t = np.linspace(0.0, 1.0, n, endpoint=False)

    clean = (
        0.8 * np.sin(2.0 * math.pi * 3.0 * t)
        + 0.35 * np.sin(2.0 * math.pi * 17.0 * t) * np.exp(-7.0 * (t - 0.35) ** 2)
        + 0.5 * (t > 0.62).astype(float)
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n)
    noisy = clean + noise

    return clean.astype(float), noisy.astype(float)


def run_demo() -> None:
    """Run deterministic MVP demonstration without interactive input."""
    clean, noisy = build_demo_signal(n=150, seed=2026, noise_std=0.25)

    padded_noisy, original_length = pad_to_power_of_two(noisy)
    levels = 4

    approx, details = haar_fwt(padded_noisy, levels=levels)
    reconstructed_full = haar_ifwt(approx, details)
    reconstructed = reconstructed_full[:original_length]

    recon_error = float(np.max(np.abs(reconstructed - noisy)))

    denoised, tau = wavelet_denoise(noisy_signal=noisy, levels=levels, threshold_scale=1.0)

    noisy_mse = mse(noisy, clean)
    denoised_mse = mse(denoised, clean)
    improvement = noisy_mse / denoised_mse if denoised_mse > 0 else float("inf")

    print("=" * 84)
    print("Haar Fast Wavelet Transform MVP")
    print("=" * 84)
    print(f"original_length={original_length}, padded_length={padded_noisy.size}, levels={levels}")
    print(f"max_abs_reconstruction_error={recon_error:.3e}")
    print(f"soft_threshold_tau={tau:.6f}")
    print(f"mse_noisy={noisy_mse:.6e}")
    print(f"mse_denoised={denoised_mse:.6e}")
    print(f"improvement_ratio={improvement:.3f}x")

    preview_k = 6
    approx_preview = ", ".join(f"{v:+.4f}" for v in approx[:preview_k])
    print(f"approx_L_preview=[{approx_preview}]")

    for i, d in enumerate(details, start=1):
        d_preview = ", ".join(f"{v:+.4f}" for v in d[:preview_k])
        print(f"detail_L{i}_preview=[{d_preview}]")


@dataclass
class _FWTQuickSanity:
    """Internal structure used only to expose core run stats in one object."""

    result: FWTResult
    mse_noisy: float
    mse_denoised: float
    threshold: float


def _build_sanity_object(clean: np.ndarray, noisy: np.ndarray, levels: int) -> _FWTQuickSanity:
    """Create one compact object to keep internal values logically grouped."""
    padded_noisy, original_length = pad_to_power_of_two(noisy)
    approx, details = haar_fwt(padded_noisy, levels=levels)
    reconstructed = haar_ifwt(approx, details)[:original_length]

    result = FWTResult(
        original_length=original_length,
        padded_length=int(padded_noisy.size),
        levels=levels,
        approx=approx,
        details=details,
        reconstructed=reconstructed,
    )

    denoised, tau = wavelet_denoise(noisy, levels=levels, threshold_scale=1.0)
    return _FWTQuickSanity(
        result=result,
        mse_noisy=mse(noisy, clean),
        mse_denoised=mse(denoised, clean),
        threshold=tau,
    )


def main() -> None:
    # Keep one internal sanity object path executed as part of the MVP to ensure
    # FWTResult dataclass remains exercised and coherent with the pipeline.
    clean, noisy = build_demo_signal(n=150, seed=2026, noise_std=0.25)
    _ = _build_sanity_object(clean, noisy, levels=4)
    run_demo()


if __name__ == "__main__":
    main()
