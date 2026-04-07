"""ARPES (Angle-Resolved Photoemission Spectroscopy) minimal runnable MVP.

The script builds a synthetic ARPES intensity map, extracts MDC peaks,
fits near-EF dispersion with scikit-learn, and uses PyTorch to refine
simple tight-binding parameters from the measured spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

K_B_EV = 8.617333262e-5  # eV / K
K_COEFF = 0.5123  # A^-1 / sqrt(eV), free-electron final-state approximation


@dataclass(frozen=True)
class ARPESParams:
    """Physical and numerical settings for the ARPES MVP."""

    hv_eV: float = 21.2
    work_function_eV: float = 4.6
    lattice_a_A: float = 3.2

    t_true_eV: float = 0.90
    mu_true_eV: float = 0.25
    gamma_true_eV: float = 0.030

    temperature_K: float = 35.0
    theta_min_deg: float = 0.0
    theta_max_deg: float = 18.0
    n_theta: int = 181

    omega_min_eV: float = -0.22
    omega_max_eV: float = 0.08
    n_omega: int = 241

    background: float = 0.015
    noise_std: float = 0.015
    sigma_omega_px: float = 1.2
    sigma_theta_px: float = 0.9

    mdc_energy_min_eV: float = -0.14
    mdc_energy_max_eV: float = -0.02
    mdc_count: int = 12

    torch_epochs: int = 500
    torch_lr: float = 0.04
    torch_weight_decay: float = 1e-4


def check_params(params: ARPESParams) -> None:
    """Validate parameter ranges."""

    if params.hv_eV <= params.work_function_eV:
        raise ValueError("hv_eV must be larger than work_function_eV")
    if params.lattice_a_A <= 0:
        raise ValueError("lattice_a_A must be positive")
    if params.t_true_eV <= 0 or params.gamma_true_eV <= 0:
        raise ValueError("t_true_eV and gamma_true_eV must be positive")
    if params.n_theta < 64 or params.n_omega < 64:
        raise ValueError("grids are too coarse")
    if params.theta_max_deg <= params.theta_min_deg:
        raise ValueError("invalid angle range")
    if params.omega_max_eV <= params.omega_min_eV:
        raise ValueError("invalid energy range")
    if params.temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if params.mdc_count < 6:
        raise ValueError("mdc_count too small for robust fit")


def fermi_dirac(omega_eV: np.ndarray, temperature_K: float) -> np.ndarray:
    """Stable Fermi distribution f(omega,T) with EF=0."""

    kbt = K_B_EV * temperature_K
    x = np.clip(omega_eV / kbt, -80.0, 80.0)
    return 1.0 / (np.exp(x) + 1.0)


def angle_to_k_parallel(theta_deg: np.ndarray, hv_eV: float, work_function_eV: float) -> np.ndarray:
    """Convert emission angle to in-plane momentum k_parallel."""

    e_kin_ref = max(hv_eV - work_function_eV, 1e-6)
    theta_rad = np.deg2rad(theta_deg)
    return K_COEFF * np.sqrt(e_kin_ref) * np.sin(theta_rad)


def tight_binding_dispersion(k_Ainv: np.ndarray, t_eV: float, mu_eV: float, lattice_a_A: float) -> np.ndarray:
    """1D nearest-neighbor tight-binding band: epsilon(k)."""

    return -2.0 * t_eV * np.cos(k_Ainv * lattice_a_A) - mu_eV


def matrix_element(k_Ainv: np.ndarray) -> np.ndarray:
    """Smooth k-dependent matrix-element modulation."""

    k_norm = (k_Ainv - k_Ainv.min()) / max(1e-12, (k_Ainv.max() - k_Ainv.min()))
    return 0.70 + 0.30 * np.cos(2.0 * np.pi * (k_norm - 0.15))


def simulate_arpes(params: ARPESParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic ARPES intensity map I(omega,theta)."""

    rng = np.random.default_rng(42)

    theta = np.linspace(params.theta_min_deg, params.theta_max_deg, params.n_theta)
    omega = np.linspace(params.omega_min_eV, params.omega_max_eV, params.n_omega)
    k = angle_to_k_parallel(theta, params.hv_eV, params.work_function_eV)

    eps = tight_binding_dispersion(k, params.t_true_eV, params.mu_true_eV, params.lattice_a_A)
    a_kw = (params.gamma_true_eV / np.pi) / (
        (omega[:, None] - eps[None, :]) ** 2 + params.gamma_true_eV**2
    )

    f_w = fermi_dirac(omega, params.temperature_K)[:, None]
    m_k = matrix_element(k)[None, :]

    intensity = m_k * a_kw * f_w + params.background
    intensity = gaussian_filter(intensity, sigma=(params.sigma_omega_px, params.sigma_theta_px), mode="nearest")

    noise_scale = params.noise_std * np.max(intensity)
    noisy = intensity + rng.normal(0.0, noise_scale, size=intensity.shape)
    noisy = np.clip(noisy, 0.0, None)

    noisy /= np.max(noisy)
    return theta, omega, k, noisy, eps


def lorentzian_with_bg(k: np.ndarray, amp: float, k0: float, gamma: float, bg: float) -> np.ndarray:
    """Single-peak Lorentzian + constant background model."""

    g2 = gamma * gamma
    return bg + amp * g2 / ((k - k0) ** 2 + g2)


def extract_mdc_peaks(
    omega: np.ndarray,
    k: np.ndarray,
    intensity_map: np.ndarray,
    params: ARPESParams,
) -> pd.DataFrame:
    """Fit MDC slices at selected energies and extract k-peak positions."""

    target_energies = np.linspace(params.mdc_energy_min_eV, params.mdc_energy_max_eV, params.mdc_count)
    rows: list[dict[str, float]] = []

    for target in target_energies:
        idx_w = int(np.argmin(np.abs(omega - target)))
        w_real = float(omega[idx_w])

        y = intensity_map[idx_w, :]
        x = k

        p0 = np.array(
            [
                float(np.max(y) - np.min(y)),
                float(x[np.argmax(y)]),
                0.030,
                float(np.min(y)),
            ]
        )

        lower = np.array([0.0, float(x.min()), 0.004, 0.0])
        upper = np.array([2.0, float(x.max()), 0.250, 1.0])

        try:
            popt, _ = curve_fit(
                lorentzian_with_bg,
                x,
                y,
                p0=p0,
                bounds=(lower, upper),
                maxfev=20000,
            )
        except RuntimeError:
            continue

        amp, k0, gamma, bg = [float(v) for v in popt]
        y_fit = lorentzian_with_bg(x, amp, k0, gamma, bg)
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / max(1e-12, ss_tot)

        rows.append(
            {
                "omega_eV": w_real,
                "k_peak_Ainv": k0,
                "gamma_mdc_Ainv": gamma,
                "amp": amp,
                "bg": bg,
                "fit_r2": r2,
            }
        )

    df = pd.DataFrame(rows).sort_values("omega_eV").reset_index(drop=True)
    return df


def fit_linear_dispersion(peaks_df: pd.DataFrame) -> dict[str, float]:
    """Fit near-EF dispersion omega = vF * k + b with linear regression."""

    x = peaks_df[["k_peak_Ainv"]].to_numpy()
    y = peaks_df["omega_eV"].to_numpy()

    reg = LinearRegression()
    reg.fit(x, y)

    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    kf = -intercept / slope
    r2 = float(reg.score(x, y))

    return {
        "vF_eV_A": slope,
        "kF_Ainv": float(kf),
        "intercept_eV": intercept,
        "fit_r2": r2,
    }


def _inv_softplus(x: float) -> float:
    """Numerically stable inverse softplus for positive x."""

    x = max(x, 1e-6)
    return float(np.log(np.expm1(x)))


def torch_refine_tb_params(
    omega: np.ndarray,
    k: np.ndarray,
    intensity_map: np.ndarray,
    params: ARPESParams,
) -> dict[str, float]:
    """Use PyTorch autograd to recover TB parameters from intensity map."""

    torch.manual_seed(42)

    # Downsample to reduce optimization cost while preserving structure.
    omega_ds = omega[::2]
    k_ds = k[::2]
    iobs_ds = intensity_map[::2, ::2]

    omega_t = torch.tensor(omega_ds, dtype=torch.float64)
    k_t = torch.tensor(k_ds, dtype=torch.float64)
    obs_t = torch.tensor(iobs_ds, dtype=torch.float64)

    raw_t = torch.nn.Parameter(torch.tensor(_inv_softplus(0.7), dtype=torch.float64))
    mu = torch.nn.Parameter(torch.tensor(0.15, dtype=torch.float64))
    raw_gamma = torch.nn.Parameter(torch.tensor(_inv_softplus(0.05), dtype=torch.float64))
    raw_scale = torch.nn.Parameter(torch.tensor(_inv_softplus(0.5), dtype=torch.float64))
    raw_bg = torch.nn.Parameter(torch.tensor(_inv_softplus(0.02), dtype=torch.float64))

    opt = torch.optim.Adam(
        [raw_t, mu, raw_gamma, raw_scale, raw_bg],
        lr=params.torch_lr,
        weight_decay=params.torch_weight_decay,
    )

    kbt = K_B_EV * params.temperature_K

    for _ in range(params.torch_epochs):
        opt.zero_grad()

        t = torch.nn.functional.softplus(raw_t)
        gamma = torch.nn.functional.softplus(raw_gamma)
        scale = torch.nn.functional.softplus(raw_scale)
        bg = torch.nn.functional.softplus(raw_bg)

        eps = -2.0 * t * torch.cos(k_t * params.lattice_a_A) - mu
        a_kw = (gamma / np.pi) / ((omega_t[:, None] - eps[None, :]) ** 2 + gamma**2)

        x_fd = torch.clamp(omega_t / kbt, min=-80.0, max=80.0)
        f_w = 1.0 / (torch.exp(x_fd) + 1.0)

        m_k = 0.70 + 0.30 * torch.cos(
            2.0
            * np.pi
            * ((k_t - torch.min(k_t)) / torch.clamp(torch.max(k_t) - torch.min(k_t), min=1e-12) - 0.15)
        )

        pred = scale * (a_kw * f_w[:, None] * m_k[None, :]) + bg
        pred = pred / torch.clamp(torch.max(pred), min=1e-12)

        loss = torch.mean((pred - obs_t) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        t_fit = float(torch.nn.functional.softplus(raw_t).item())
        mu_fit = float(mu.item())
        gamma_fit = float(torch.nn.functional.softplus(raw_gamma).item())
        scale_fit = float(torch.nn.functional.softplus(raw_scale).item())
        bg_fit = float(torch.nn.functional.softplus(raw_bg).item())

        eps = -2.0 * torch.nn.functional.softplus(raw_t) * torch.cos(k_t * params.lattice_a_A) - mu
        gamma = torch.nn.functional.softplus(raw_gamma)
        a_kw = (gamma / np.pi) / ((omega_t[:, None] - eps[None, :]) ** 2 + gamma**2)

        x_fd = torch.clamp(omega_t / kbt, min=-80.0, max=80.0)
        f_w = 1.0 / (torch.exp(x_fd) + 1.0)

        m_k = 0.70 + 0.30 * torch.cos(
            2.0
            * np.pi
            * ((k_t - torch.min(k_t)) / torch.clamp(torch.max(k_t) - torch.min(k_t), min=1e-12) - 0.15)
        )
        pred = torch.nn.functional.softplus(raw_scale) * (a_kw * f_w[:, None] * m_k[None, :]) + torch.nn.functional.softplus(
            raw_bg
        )
        pred = pred / torch.clamp(torch.max(pred), min=1e-12)
        final_loss = float(torch.mean((pred - obs_t) ** 2).item())

    return {
        "t_fit_eV": t_fit,
        "mu_fit_eV": mu_fit,
        "gamma_fit_eV": gamma_fit,
        "scale_fit": scale_fit,
        "bg_fit": bg_fit,
        "final_mse": final_loss,
    }


def build_long_table(theta: np.ndarray, omega: np.ndarray, k: np.ndarray, intensity_map: np.ndarray) -> pd.DataFrame:
    """Flatten spectrum to a tidy table for downstream analysis/export."""

    ww, tt = np.meshgrid(omega, theta, indexing="ij")
    kk = np.tile(k[None, :], (omega.size, 1))

    return pd.DataFrame(
        {
            "omega_eV": ww.ravel(),
            "theta_deg": tt.ravel(),
            "k_Ainv": kk.ravel(),
            "intensity": intensity_map.ravel(),
        }
    )


def build_summary(
    params: ARPESParams,
    peaks_df: pd.DataFrame,
    linear_fit: dict[str, float],
    torch_fit: dict[str, float],
    k_grid: np.ndarray,
) -> pd.DataFrame:
    """Collect key diagnostics in a small summary table."""

    return pd.DataFrame(
        [
            {"metric": "n_mdc_points", "value": f"{len(peaks_df)}"},
            {"metric": "mean_mdc_fit_r2", "value": f"{peaks_df['fit_r2'].mean():.6f}"},
            {"metric": "linear_fit_r2", "value": f"{linear_fit['fit_r2']:.6f}"},
            {"metric": "kF_from_linear_fit_Ainv", "value": f"{linear_fit['kF_Ainv']:.6f}"},
            {"metric": "vF_from_linear_fit_eV_A", "value": f"{linear_fit['vF_eV_A']:.6f}"},
            {"metric": "torch_t_fit_eV", "value": f"{torch_fit['t_fit_eV']:.6f}"},
            {"metric": "torch_mu_fit_eV", "value": f"{torch_fit['mu_fit_eV']:.6f}"},
            {"metric": "torch_gamma_fit_eV", "value": f"{torch_fit['gamma_fit_eV']:.6f}"},
            {"metric": "torch_final_mse", "value": f"{torch_fit['final_mse']:.6e}"},
            {
                "metric": "kF_in_grid",
                "value": f"{float(k_grid.min()) < linear_fit['kF_Ainv'] < float(k_grid.max())}",
            },
            {
                "metric": "t_relative_error",
                "value": f"{abs(torch_fit['t_fit_eV'] - params.t_true_eV) / params.t_true_eV:.3e}",
            },
            {
                "metric": "mu_absolute_error_eV",
                "value": f"{abs(torch_fit['mu_fit_eV'] - params.mu_true_eV):.3e}",
            },
        ]
    )


def main() -> None:
    params = ARPESParams()
    check_params(params)

    theta, omega, k, intensity_map, _eps_true = simulate_arpes(params)
    peaks_df = extract_mdc_peaks(omega, k, intensity_map, params)

    if len(peaks_df) < 6:
        raise AssertionError("Not enough MDC fits succeeded for dispersion fitting")

    linear_fit = fit_linear_dispersion(peaks_df)
    torch_fit = torch_refine_tb_params(omega, k, intensity_map, params)

    long_df = build_long_table(theta, omega, k, intensity_map)
    summary_df = build_summary(params, peaks_df, linear_fit, torch_fit, k)

    print("=== ARPES MVP: synthetic spectrum -> MDC -> dispersion -> inversion ===")
    print(
        "params:",
        {
            "hv_eV": params.hv_eV,
            "work_function_eV": params.work_function_eV,
            "temperature_K": params.temperature_K,
            "theta_range_deg": (params.theta_min_deg, params.theta_max_deg),
            "omega_range_eV": (params.omega_min_eV, params.omega_max_eV),
            "grid_shape": (params.n_omega, params.n_theta),
            "true_tb": {
                "t_eV": params.t_true_eV,
                "mu_eV": params.mu_true_eV,
                "gamma_eV": params.gamma_true_eV,
            },
        },
    )

    print("\n[summary]")
    print(summary_df.to_string(index=False))

    print("\n[mdc_peaks_head]")
    print(peaks_df.head(8).to_string(index=False))

    print("\n[spectrum_table_head]")
    print(long_df.head(8).to_string(index=False))

    # Minimal quality gates for automated validation.
    if float(peaks_df["fit_r2"].mean()) < 0.85:
        raise AssertionError("Average MDC fit quality is too low")

    if not (float(k.min()) < float(linear_fit["kF_Ainv"]) < float(k.max())):
        raise AssertionError("Estimated kF is outside sampled momentum window")

    if float(linear_fit["fit_r2"]) < 0.95:
        raise AssertionError("Linear near-EF dispersion fit is too poor")

    if float(torch_fit["final_mse"]) > 1.2e-2:
        raise AssertionError("Torch inversion loss is too high")

    if abs(float(torch_fit["t_fit_eV"]) - params.t_true_eV) > 0.25:
        raise AssertionError("Recovered t is far from true value")

    # Cross-check true dispersion at extracted k points (sanity, not a strict fit target).
    eps_interp = tight_binding_dispersion(peaks_df["k_peak_Ainv"].to_numpy(), params.t_true_eV, params.mu_true_eV, params.lattice_a_A)
    mae = float(np.mean(np.abs(eps_interp - peaks_df["omega_eV"].to_numpy())))
    if mae > 0.06:
        raise AssertionError("Extracted peaks are not consistent with true-band energy scale")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
