"""Minimal runnable MVP for gravitational-wave matched filtering.

This script builds a synthetic inspiral waveform using the leading-order
(Newtonian) post-Newtonian chirp relation, injects it into colored Gaussian
noise, and recovers the signal with a transparent template-bank matched filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Physical constants (SI)
G = 6.67430e-11
C = 299_792_458.0
M_SUN = 1.98847e30
EPS = 1e-12


@dataclass(frozen=True)
class ChirpConfig:
    fs_hz: int = 1024
    duration_s: float = 16.0
    f_low_hz: float = 28.0
    f_high_hz: float = 350.0

    @property
    def n_samples(self) -> int:
        return int(self.fs_hz * self.duration_s)

    @property
    def dt(self) -> float:
        return 1.0 / self.fs_hz


@dataclass(frozen=True)
class TemplateMatch:
    mchirp_solar: float
    peak_z: float
    lag_s: float
    estimated_tc_s: float


def chirp_frequency_prefactor(mchirp_solar: float) -> float:
    """Return A in f(t)=A*(tc-t)^(-3/8) for leading-order inspiral."""
    mchirp_si = mchirp_solar * M_SUN
    return (1.0 / np.pi) * (5.0 / 256.0) ** (3.0 / 8.0) * (G * mchirp_si / C**3) ** (-5.0 / 8.0)


def generate_newtonian_inspiral(
    cfg: ChirpConfig,
    mchirp_solar: float,
    tc_s: float,
    phi_c: float,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a restricted inspiral chirp waveform.

    Returns
    -------
    t : time grid
    h : strain-like waveform (dimensionless scale)
    f : instantaneous GW frequency used to build phase
    """
    n = cfg.n_samples
    t = np.arange(n, dtype=float) * cfg.dt

    tau = np.maximum(tc_s - t, 1e-9)
    prefactor = chirp_frequency_prefactor(mchirp_solar)
    f = prefactor * tau ** (-3.0 / 8.0)

    in_band = (t < tc_s) & (f >= cfg.f_low_hz) & (f <= cfg.f_high_hz)

    # Integrating f(t) analytically gives a closed-form phase at Newtonian order.
    phase = phi_c - (16.0 / 5.0) * (np.pi * prefactor) * tau ** (5.0 / 8.0)

    h = np.zeros(n, dtype=float)
    h[in_band] = amplitude * (f[in_band] / 100.0) ** (2.0 / 3.0) * np.cos(phase[in_band])
    return t, h, f


def detector_psd_model(freq_hz: np.ndarray) -> np.ndarray:
    """A simple analytic one-sided PSD shape (LIGO-like toy curve)."""
    f = np.maximum(freq_hz, 1.0)
    s0 = 1e-46
    return s0 * ((25.0 / f) ** 4 + 2.0 + (f / 250.0) ** 2)


def generate_colored_gaussian_noise(cfg: ChirpConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate time-domain colored Gaussian noise from target one-sided PSD."""
    n = cfg.n_samples
    freqs = np.fft.rfftfreq(n, d=cfg.dt)
    psd = detector_psd_model(freqs)

    spectrum = np.zeros(freqs.size, dtype=np.complex128)

    # Positive-frequency bins excluding DC and Nyquist.
    pos_end = freqs.size - 1 if n % 2 == 0 else freqs.size
    idx = slice(1, pos_end)
    scale = np.sqrt(0.25 * n * cfg.fs_hz * psd[idx])
    spectrum[idx] = (rng.normal(size=scale.size) + 1j * rng.normal(size=scale.size)) * scale

    # Nyquist bin must be purely real for real-valued irfft reconstruction.
    if n % 2 == 0:
        spectrum[-1] = rng.normal() * np.sqrt(0.5 * n * cfg.fs_hz * psd[-1])

    noise = np.fft.irfft(spectrum, n=n)
    return noise, psd, freqs


def matched_filter_zscore(
    data: np.ndarray,
    template: np.ndarray,
    psd: np.ndarray,
    cfg: ChirpConfig,
) -> tuple[np.ndarray, float, float]:
    """Compute PSD-weighted matched filter output and convert to z-score series."""
    if data.shape != template.shape:
        raise ValueError("data and template must have the same shape")

    n = data.size
    df = cfg.fs_hz / n
    freqs = np.fft.rfftfreq(n, d=cfg.dt)

    # Windowing reduces FFT boundary artifacts for this finite-length MVP segment.
    window = np.hanning(n)
    data_fft = np.fft.rfft((data - np.mean(data)) * window)
    temp_fft = np.fft.rfft((template - np.mean(template)) * window)

    inv_psd = np.zeros_like(psd)
    valid = (freqs >= cfg.f_low_hz) & (freqs <= cfg.f_high_hz) & (psd > 0.0)
    inv_psd[valid] = 1.0 / psd[valid]

    # Complex matched-filter time series z(t) ~ IFFT[d(f) h*(f) / S_n(f)].
    z = np.fft.irfft(data_fft * np.conj(temp_fft) * inv_psd, n=n) * (4.0 * df * n)

    # Normalize with empirical background spread so thresholding is easy to read.
    z_centered = z - np.mean(z)
    z_std = max(float(np.std(z_centered)), EPS)
    z_score = z_centered / z_std

    peak_idx = int(np.argmax(np.abs(z_score)))
    signed_lag_idx = peak_idx if peak_idx <= n // 2 else peak_idx - n
    lag_s = signed_lag_idx * cfg.dt

    return z_score, float(np.max(np.abs(z_score))), float(lag_s)


def run_template_bank(
    data: np.ndarray,
    psd: np.ndarray,
    cfg: ChirpConfig,
    mchirp_bank: np.ndarray,
    template_tc_s: float,
) -> list[TemplateMatch]:
    """Evaluate matched-filter peak statistic for each chirp-mass template."""
    results: list[TemplateMatch] = []
    for mc in mchirp_bank:
        _, template, _ = generate_newtonian_inspiral(
            cfg=cfg,
            mchirp_solar=float(mc),
            tc_s=template_tc_s,
            phi_c=0.0,
            amplitude=1.0,
        )
        _, peak_z, lag_s = matched_filter_zscore(data, template, psd, cfg)
        results.append(
            TemplateMatch(
                mchirp_solar=float(mc),
                peak_z=peak_z,
                lag_s=lag_s,
                estimated_tc_s=template_tc_s + lag_s,
            )
        )
    return results


def to_dataframe(matches: list[TemplateMatch]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mchirp_solar": [m.mchirp_solar for m in matches],
            "peak_z": [m.peak_z for m in matches],
            "lag_s": [m.lag_s for m in matches],
            "estimated_tc_s": [m.estimated_tc_s for m in matches],
        }
    ).sort_values("peak_z", ascending=False, ignore_index=True)


def main() -> None:
    cfg = ChirpConfig(fs_hz=1024, duration_s=16.0, f_low_hz=28.0, f_high_hz=350.0)
    rng = np.random.default_rng(123)

    # Ground-truth injection (unknown to the bank in real searches).
    true_mchirp = 1.22
    true_tc_s = 12.7
    true_phi_c = 0.4
    true_amp = 8e-23

    # Bank uses a reference coalescence time; lag from matched filtering corrects it.
    template_tc_s = 11.0
    mchirp_bank = np.array([1.12, 1.16, 1.20, 1.22, 1.24, 1.28, 1.32], dtype=float)

    noise, psd, _freqs = generate_colored_gaussian_noise(cfg, rng)
    _, signal, inst_freq = generate_newtonian_inspiral(
        cfg=cfg,
        mchirp_solar=true_mchirp,
        tc_s=true_tc_s,
        phi_c=true_phi_c,
        amplitude=true_amp,
    )
    data = noise + signal

    signal_rms = float(np.sqrt(np.mean(signal**2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    injected_in_band = int(np.count_nonzero(np.abs(signal) > 0.0))

    data_matches = run_template_bank(data, psd, cfg, mchirp_bank, template_tc_s)
    noise_matches = run_template_bank(noise, psd, cfg, mchirp_bank, template_tc_s)

    data_df = to_dataframe(data_matches)
    noise_df = to_dataframe(noise_matches)

    best = data_df.iloc[0]
    second = data_df.iloc[1]
    best_mc = float(best["mchirp_solar"])
    best_peak_z = float(best["peak_z"])
    best_tc = float(best["estimated_tc_s"])

    mass_abs_err = abs(best_mc - true_mchirp)
    tc_abs_err = abs(best_tc - true_tc_s)
    background_peak = float(noise_df["peak_z"].max())
    contrast = best_peak_z / max(float(second["peak_z"]), EPS)

    checks = {
        "best peak z >= 8": best_peak_z >= 8.0,
        "chirp mass abs error <= 0.03": mass_abs_err <= 0.03,
        "coalescence-time abs error <= 0.15 s": tc_abs_err <= 0.15,
        "best peak exceeds noise-only peak by >= 4": best_peak_z >= background_peak + 4.0,
        "best/second peak contrast >= 1.8": contrast >= 1.8,
    }

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("=== Gravitational Wave Matched-Filter MVP (PHYS-0371) ===")
    print(f"samples={cfg.n_samples}, fs={cfg.fs_hz} Hz, duration={cfg.duration_s:.2f} s")
    print(f"band=[{cfg.f_low_hz:.1f}, {cfg.f_high_hz:.1f}] Hz")
    print(
        "injection: "
        f"mchirp={true_mchirp:.3f} Msun, tc={true_tc_s:.3f} s, "
        f"signal_rms={signal_rms:.3e}, noise_rms={noise_rms:.3e}, active_samples={injected_in_band}"
    )

    print("\nTemplate-bank results on injected data (sorted):")
    print(data_df.to_string(index=False))

    print("\nNoise-only baseline (same bank):")
    print(noise_df.to_string(index=False))

    print("\nRecovered parameters:")
    print(f"best mchirp = {best_mc:.3f} Msun (abs err={mass_abs_err:.4f})")
    print(f"best tc     = {best_tc:.4f} s (abs err={tc_abs_err:.4f} s)")
    print(f"best peak z = {best_peak_z:.3f}, noise-only max z = {background_peak:.3f}, contrast = {contrast:.3f}")

    print("\nSanity checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    # Extra consistency check: injected instantaneous frequency should stay finite where active.
    finite_ok = np.all(np.isfinite(inst_freq[(inst_freq >= cfg.f_low_hz) & (inst_freq <= cfg.f_high_hz)]))
    print(f"- finite instantaneous frequency in band: {'OK' if finite_ok else 'FAIL'}")

    all_ok = all(checks.values()) and finite_ok
    print(f"\nValidation: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
