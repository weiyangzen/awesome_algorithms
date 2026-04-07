"""Minimal runnable MVP for X-ray absorption spectroscopy (XAS/XAFS).

Pipeline implemented in source code (no black-box EXAFS package):
1) Generate a synthetic mu(E) spectrum with known EXAFS shells.
2) Estimate absorption edge E0 from first derivative.
3) Perform pre-edge subtraction and edge-step normalization.
4) Convert E -> k and build chi(k).
5) Apply k^2 weighting and a Hanning window.
6) Compute |FT[chi(k)]| in R-space via explicit matrix transform.
7) Detect shell peaks and validate against known shell distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

# hbar^2 / (2m_e) in eV*Angstrom^2
K_CONVERSION_EV_A2 = 3.80998212


@dataclass
class Shell:
    coordination: float
    distance_a: float
    sigma2_a2: float
    amplitude: float
    phase_shift_rad: float


@dataclass
class SyntheticConfig:
    e0_true_ev: float = 7112.0
    energy_min_ev: float = 7050.0
    energy_max_ev: float = 7600.0
    num_energy_points: int = 2400
    noise_std: float = 0.002
    rng_seed: int = 20260407


@dataclass
class PipelineConfig:
    pre_edge_range_ev: tuple[float, float] = (-120.0, -30.0)
    post_edge_range_ev: tuple[float, float] = (150.0, 320.0)
    exafs_start_offset_ev: float = 5.0
    k_interp_min_a_inv: float = 2.5
    k_interp_max_a_inv: float = 12.5
    k_step_a_inv: float = 0.05
    k_window_range_a_inv: tuple[float, float] = (3.0, 11.0)
    r_min_a: float = 0.5
    r_max_a: float = 4.5
    r_num: int = 801


@dataclass
class ValidationReport:
    e0_true_ev: float
    e0_est_ev: float
    e0_abs_error_ev: float
    true_shells_a: tuple[float, float]
    est_shells_a: tuple[float, float]
    shell_abs_errors_a: tuple[float, float]


def exafs_chi_model(k: np.ndarray, shells: list[Shell]) -> np.ndarray:
    """Build synthetic EXAFS chi(k) from shell contributions.

    chi(k) = sum_j [ Nj * Aj / (k * Rj^2) * exp(-2 sigma_j^2 k^2) * exp(-2Rj/lambda(k))
                     * sin(2kRj + phase_j) ]
    """
    safe_k = np.maximum(k, 1e-6)
    lambda_k = 8.0 + 0.35 * safe_k  # simple mean free path model in Angstrom

    chi = np.zeros_like(safe_k)
    for shell in shells:
        prefactor = shell.coordination * shell.amplitude / (safe_k * shell.distance_a**2)
        damping = np.exp(-2.0 * shell.sigma2_a2 * safe_k * safe_k) * np.exp(
            -2.0 * shell.distance_a / lambda_k
        )
        oscillation = np.sin(2.0 * safe_k * shell.distance_a + shell.phase_shift_rad)
        chi += prefactor * damping * oscillation

    return chi


def generate_synthetic_xas(cfg: SyntheticConfig, shells: list[Shell]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return energy grid, mu(E), k_true(E), chi_true(E)."""
    energy = np.linspace(cfg.energy_min_ev, cfg.energy_max_ev, cfg.num_energy_points)
    delta_e = np.maximum(energy - cfg.e0_true_ev, 0.0)
    k_true = np.sqrt(delta_e / K_CONVERSION_EV_A2)
    chi_true = exafs_chi_model(k_true, shells)

    # Slowly varying atomic background + soft edge step.
    pre_line = 0.20 + 2.0e-4 * (energy - cfg.e0_true_ev)
    step_height = 1.15
    step_soft = 1.0 / (1.0 + np.exp(-(energy - cfg.e0_true_ev) / 1.2))

    mu_ideal = pre_line + step_height * step_soft * (1.0 + chi_true)

    rng = np.random.default_rng(cfg.rng_seed)
    noise = rng.normal(loc=0.0, scale=cfg.noise_std, size=energy.shape)
    mu_noisy = mu_ideal + noise

    return energy, mu_noisy, k_true, chi_true


def estimate_e0(energy: np.ndarray, mu: np.ndarray) -> float:
    window = 31 if energy.size >= 31 else (energy.size // 2) * 2 + 1
    mu_smooth = savgol_filter(mu, window_length=window, polyorder=3)
    dmu_de = np.gradient(mu_smooth, energy)
    idx = int(np.argmax(dmu_de))
    return float(energy[idx])


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coef = np.polyfit(x, y, deg=1)
    slope, intercept = float(coef[0]), float(coef[1])
    return slope, intercept


def normalize_xas(energy: np.ndarray, mu: np.ndarray, e0_est: float, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, float]:
    pre_lo, pre_hi = cfg.pre_edge_range_ev
    post_lo, post_hi = cfg.post_edge_range_ev

    pre_mask = (energy >= e0_est + pre_lo) & (energy <= e0_est + pre_hi)
    post_mask = (energy >= e0_est + post_lo) & (energy <= e0_est + post_hi)

    if pre_mask.sum() < 30 or post_mask.sum() < 30:
        raise RuntimeError("Not enough points for pre-edge/post-edge line fit.")

    pre_slope, pre_intercept = _fit_line(energy[pre_mask], mu[pre_mask])
    post_slope, post_intercept = _fit_line(energy[post_mask], mu[post_mask])

    pre_line = pre_slope * energy + pre_intercept
    edge_step = (post_slope * e0_est + post_intercept) - (pre_slope * e0_est + pre_intercept)
    if edge_step <= 1e-6:
        raise RuntimeError("Estimated edge step is non-positive.")

    mu_norm = (mu - pre_line) / edge_step
    chi_e = mu_norm - 1.0
    return mu_norm, chi_e, float(edge_step)


def energy_to_k(energy: np.ndarray, e0: float, chi_e: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    mask = energy > (e0 + cfg.exafs_start_offset_ev)
    if mask.sum() < 100:
        raise RuntimeError("Not enough EXAFS points above edge.")

    k_nonuniform = np.sqrt((energy[mask] - e0) / K_CONVERSION_EV_A2)
    chi_nonuniform = chi_e[mask]

    k_max_allowed = min(cfg.k_interp_max_a_inv, float(np.max(k_nonuniform)) - 0.05)
    if k_max_allowed <= cfg.k_interp_min_a_inv + 0.2:
        raise RuntimeError("k-range too narrow for interpolation.")

    k_uniform = np.arange(cfg.k_interp_min_a_inv, k_max_allowed, cfg.k_step_a_inv)
    if k_uniform.size < 80:
        raise RuntimeError("Interpolated k-grid has too few samples.")

    interpolator = interp1d(k_nonuniform, chi_nonuniform, kind="linear", bounds_error=True)
    chi_uniform = interpolator(k_uniform)
    return k_uniform, chi_uniform


def make_hanning_window(k: np.ndarray, k_min: float, k_max: float) -> np.ndarray:
    win = np.zeros_like(k)
    mask = (k >= k_min) & (k <= k_max)
    n = int(mask.sum())
    if n < 3:
        raise RuntimeError("Window support too small. Adjust k window range.")
    win[mask] = np.hanning(n)
    return win


def exafs_fourier_transform(
    k: np.ndarray,
    chi_k: np.ndarray,
    k_weight_power: int,
    k_window_range: tuple[float, float],
    r_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weighted = chi_k * (k**k_weight_power)
    window = make_hanning_window(k, k_window_range[0], k_window_range[1])
    signal = weighted * window

    dk = float(k[1] - k[0])
    phase = np.exp(2.0j * np.outer(r_grid, k))
    ft_complex = phase @ signal * dk
    ft_mag = np.abs(ft_complex)
    return ft_mag, signal


def pick_shell_peaks(r: np.ndarray, ft_mag: np.ndarray, top_n: int = 2) -> np.ndarray:
    max_ft = float(np.max(ft_mag))
    peaks = np.array([], dtype=int)
    props: dict[str, np.ndarray] = {}
    for frac in (0.10, 0.07, 0.05, 0.03, 0.02):
        peaks, props = find_peaks(ft_mag, prominence=frac * max_ft, distance=20)
        if peaks.size >= top_n:
            break
    if peaks.size < top_n:
        raise RuntimeError("Not enough R-space peaks detected, even after adaptive prominence fallback.")

    heights = props["prominences"]
    order = np.argsort(heights)[::-1]
    chosen = peaks[order[:top_n]]
    chosen_r = np.sort(r[chosen])
    return chosen_r


def validate_results(e0_true: float, e0_est: float, true_shells: tuple[float, float], est_shells: tuple[float, float]) -> ValidationReport:
    e0_error = abs(e0_est - e0_true)
    shell_errors = (abs(est_shells[0] - true_shells[0]), abs(est_shells[1] - true_shells[1]))
    return ValidationReport(
        e0_true_ev=e0_true,
        e0_est_ev=e0_est,
        e0_abs_error_ev=e0_error,
        true_shells_a=true_shells,
        est_shells_a=est_shells,
        shell_abs_errors_a=shell_errors,
    )


def main() -> None:
    synth_cfg = SyntheticConfig()
    pipe_cfg = PipelineConfig()

    true_shell_list = [
        Shell(coordination=4.0, distance_a=1.95, sigma2_a2=0.0038, amplitude=1.00, phase_shift_rad=0.15),
        Shell(coordination=2.0, distance_a=3.10, sigma2_a2=0.0058, amplitude=0.85, phase_shift_rad=0.55),
    ]
    true_shells = (true_shell_list[0].distance_a, true_shell_list[1].distance_a)

    energy, mu, _k_true, _chi_true = generate_synthetic_xas(synth_cfg, true_shell_list)
    e0_est = estimate_e0(energy, mu)

    mu_norm, chi_e, edge_step = normalize_xas(energy, mu, e0_est, pipe_cfg)
    k_uniform, chi_uniform = energy_to_k(energy, e0_est, chi_e, pipe_cfg)

    r_grid = np.linspace(pipe_cfg.r_min_a, pipe_cfg.r_max_a, pipe_cfg.r_num)
    ft_mag, windowed_signal = exafs_fourier_transform(
        k_uniform,
        chi_uniform,
        k_weight_power=2,
        k_window_range=pipe_cfg.k_window_range_a_inv,
        r_grid=r_grid,
    )

    est_shells_array = pick_shell_peaks(r_grid, ft_mag, top_n=2)
    est_shells = (float(est_shells_array[0]), float(est_shells_array[1]))

    report = validate_results(
        e0_true=synth_cfg.e0_true_ev,
        e0_est=e0_est,
        true_shells=true_shells,
        est_shells=est_shells,
    )

    checks = {
        "|E0_est - E0_true| < 6 eV": report.e0_abs_error_ev < 6.0,
        "|R1_est - R1_true| < 0.30 A": report.shell_abs_errors_a[0] < 0.30,
        "|R2_est - R2_true| < 0.35 A": report.shell_abs_errors_a[1] < 0.35,
    }

    k_preview = pd.DataFrame(
        {
            "k_A^-1": k_uniform,
            "chi(k)": chi_uniform,
            "k^2*chi(k)*window": windowed_signal,
        }
    )
    r_preview = pd.DataFrame({"R_A": r_grid, "|FT|": ft_mag})

    print("=== XAS/XAFS MVP (PHYS-0465) ===")
    print("Synthetic Fe-like K-edge EXAFS pipeline with explicit preprocessing + FT")

    print("\n[Edge and normalization]")
    print(
        "E0_true = {e0t:.3f} eV, E0_est = {e0e:.3f} eV, |err| = {eerr:.3f} eV, edge_step = {step:.4f}".format(
            e0t=report.e0_true_ev,
            e0e=report.e0_est_ev,
            eerr=report.e0_abs_error_ev,
            step=edge_step,
        )
    )

    print("\n[R-space shell validation]")
    print(
        "R1_true = {r1t:.3f} A, R1_est = {r1e:.3f} A, |err| = {r1err:.3f} A".format(
            r1t=report.true_shells_a[0],
            r1e=report.est_shells_a[0],
            r1err=report.shell_abs_errors_a[0],
        )
    )
    print(
        "R2_true = {r2t:.3f} A, R2_est = {r2e:.3f} A, |err| = {r2err:.3f} A".format(
            r2t=report.true_shells_a[1],
            r2e=report.est_shells_a[1],
            r2err=report.shell_abs_errors_a[1],
        )
    )

    print("\n[Preview: k-space samples]")
    print(k_preview.head(8).to_string(index=False))

    top_r_idx = np.argsort(r_preview["|FT|"].values)[-8:]
    top_r = r_preview.iloc[np.sort(top_r_idx)]
    print("\n[Preview: strongest R-space bins]")
    print(top_r.to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
        return

    print("\nValidation: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
