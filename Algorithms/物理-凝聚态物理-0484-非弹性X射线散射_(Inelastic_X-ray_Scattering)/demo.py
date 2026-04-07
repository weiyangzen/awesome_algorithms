"""Minimal runnable MVP for Inelastic X-ray Scattering (IXS).

This script builds a synthetic IXS map S(Q, w), performs per-Q spectrum
fitting to recover phonon energies and linewidths, regresses the dispersion,
and runs a PyTorch global refinement of key physical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

KB_MEV_PER_K = 8.617333262e-2  # meV/K


@dataclass(frozen=True)
class IXSParams:
    """Physical and numerical settings for the IXS MVP."""

    temperature_K: float = 280.0

    q_min_Ainv: float = 0.7
    q_max_Ainv: float = 3.2
    n_q: int = 24

    omega_min_meV: float = -35.0
    omega_max_meV: float = 35.0
    n_omega: int = 351

    resolution_sigma_meV: float = 1.15

    sound_velocity_true_meV_A: float = 11.0
    gap_true_meV: float = 4.8
    gamma0_true_meV: float = 1.3
    gamma2_true_meV_A2: float = 0.34

    amp_scale_true: float = 950.0
    elastic_scale_true: float = 150.0
    background_true: float = 1.15

    random_seed: int = 20260407

    torch_epochs: int = 420
    torch_lr: float = 0.05
    torch_weight_decay: float = 2e-4


def check_params(params: IXSParams) -> None:
    """Validate parameter ranges."""

    if params.temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if params.n_q < 8 or params.n_omega < 128:
        raise ValueError("grid is too coarse for stable fitting")
    if params.q_max_Ainv <= params.q_min_Ainv:
        raise ValueError("invalid q range")
    if params.omega_max_meV <= params.omega_min_meV:
        raise ValueError("invalid omega range")
    if params.resolution_sigma_meV <= 0.0:
        raise ValueError("resolution_sigma_meV must be positive")
    if params.sound_velocity_true_meV_A <= 0.0:
        raise ValueError("sound_velocity_true_meV_A must be positive")
    if params.gap_true_meV <= 0.0:
        raise ValueError("gap_true_meV must be positive")
    if params.gamma0_true_meV <= 0.0 or params.gamma2_true_meV_A2 <= 0.0:
        raise ValueError("damping coefficients must be positive")


def bose_occupation(energy_meV: np.ndarray | float, temperature_K: float) -> np.ndarray:
    """Return Bose-Einstein occupation n(E,T) with numerical clipping."""

    energy = np.asarray(energy_meV, dtype=float)
    x = np.clip(energy / (KB_MEV_PER_K * temperature_K), 1e-8, 80.0)
    return 1.0 / (np.exp(x) - 1.0)


def lorentzian(omega_meV: np.ndarray, center_meV: float, gamma_meV: float) -> np.ndarray:
    """Area-normalized Lorentzian profile."""

    gamma = max(float(gamma_meV), 1e-6)
    return (gamma / np.pi) / ((omega_meV - center_meV) ** 2 + gamma**2)


def phonon_dispersion(q_Ainv: np.ndarray | float, c_meV_A: float, gap_meV: float) -> np.ndarray:
    """Gapped acoustic-like dispersion: Omega(Q)=sqrt((cQ)^2+gap^2)."""

    q = np.asarray(q_Ainv, dtype=float)
    return np.sqrt((c_meV_A * q) ** 2 + gap_meV**2)


def phonon_damping(q_Ainv: np.ndarray | float, g0_meV: float, g2_meV_A2: float) -> np.ndarray:
    """Quadratic linewidth model Gamma(Q)=g0+g2*Q^2."""

    q = np.asarray(q_Ainv, dtype=float)
    return g0_meV + g2_meV_A2 * q**2


def q_amplitude(q_Ainv: float, amp_scale: float) -> float:
    """Empirical inelastic intensity envelope versus Q."""

    return float(amp_scale * q_Ainv**2 / (1.0 + (q_Ainv / 2.0) ** 2))


def q_elastic(q_Ainv: float, elastic_scale: float) -> float:
    """Empirical elastic line envelope versus Q."""

    return float(elastic_scale * np.exp(-q_Ainv / 2.6))


def unresolved_ixs_spectrum(
    omega_meV: np.ndarray,
    omega0_meV: float,
    gamma_meV: float,
    amp: float,
    elastic_amp: float,
    background: float,
    temperature_K: float,
) -> np.ndarray:
    """Build a one-phonon IXS spectrum before instrumental broadening."""

    n_bose = float(bose_occupation(omega0_meV, temperature_K))
    stokes = (n_bose + 1.0) * lorentzian(omega_meV, omega0_meV, gamma_meV)
    anti_stokes = n_bose * lorentzian(omega_meV, -omega0_meV, gamma_meV)
    elastic = elastic_amp * lorentzian(omega_meV, 0.0, 0.2)
    return amp * (stokes + anti_stokes) + elastic + background


def resolved_ixs_spectrum(
    omega_meV: np.ndarray,
    omega0_meV: float,
    gamma_meV: float,
    amp: float,
    elastic_amp: float,
    background: float,
    temperature_K: float,
    resolution_sigma_meV: float,
) -> np.ndarray:
    """Apply Gaussian instrument resolution to the unresolved spectrum."""

    raw = unresolved_ixs_spectrum(
        omega_meV=omega_meV,
        omega0_meV=omega0_meV,
        gamma_meV=gamma_meV,
        amp=amp,
        elastic_amp=elastic_amp,
        background=background,
        temperature_K=temperature_K,
    )
    d_omega = float(omega_meV[1] - omega_meV[0])
    sigma_px = resolution_sigma_meV / d_omega
    return gaussian_filter1d(raw, sigma=sigma_px, mode="nearest")


def simulate_ixs_map(params: IXSParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic IXS map: expected counts and noisy observations."""

    q = np.linspace(params.q_min_Ainv, params.q_max_Ainv, params.n_q)
    omega = np.linspace(params.omega_min_meV, params.omega_max_meV, params.n_omega)

    expected = np.zeros((params.n_q, params.n_omega), dtype=float)

    for iq, qv in enumerate(q):
        omega0 = float(phonon_dispersion(qv, params.sound_velocity_true_meV_A, params.gap_true_meV))
        gamma = float(phonon_damping(qv, params.gamma0_true_meV, params.gamma2_true_meV_A2))
        amp = q_amplitude(qv, params.amp_scale_true)
        elastic_amp = q_elastic(qv, params.elastic_scale_true)

        expected[iq, :] = resolved_ixs_spectrum(
            omega_meV=omega,
            omega0_meV=omega0,
            gamma_meV=gamma,
            amp=amp,
            elastic_amp=elastic_amp,
            background=params.background_true,
            temperature_K=params.temperature_K,
            resolution_sigma_meV=params.resolution_sigma_meV,
        )

    expected = np.clip(expected, 1e-9, None)
    observed = np.random.default_rng(params.random_seed + 1).poisson(expected).astype(float)
    return q, omega, expected, observed


def fit_single_q_spectrum(
    omega_meV: np.ndarray,
    counts: np.ndarray,
    temperature_K: float,
    resolution_sigma_meV: float,
) -> dict[str, float | bool]:
    """Fit one IXS spectrum at fixed Q with constrained nonlinear least squares."""

    tail = np.concatenate([counts[:20], counts[-20:]])
    bg0 = float(np.median(tail))

    peak_idx = int(np.argmax(counts))
    omega_guess = float(np.clip(abs(omega_meV[peak_idx]), 4.0, 32.0))

    p0 = np.array([omega_guess, 2.0, 800.0, 120.0, max(bg0, 0.2)], dtype=float)
    lower = np.array([2.0, 0.2, 1.0, 0.0, 0.0], dtype=float)
    upper = np.array([45.0, 12.0, 5e4, 5e3, max(float(np.max(counts) * 2.0), 10.0)], dtype=float)

    def model(w: np.ndarray, omega0: float, gamma: float, amp: float, elastic_amp: float, bg: float) -> np.ndarray:
        return resolved_ixs_spectrum(
            omega_meV=w,
            omega0_meV=omega0,
            gamma_meV=gamma,
            amp=amp,
            elastic_amp=elastic_amp,
            background=bg,
            temperature_K=temperature_K,
            resolution_sigma_meV=resolution_sigma_meV,
        )

    try:
        popt, _ = curve_fit(
            model,
            omega_meV,
            counts,
            p0=p0,
            bounds=(lower, upper),
            maxfev=22000,
        )
    except RuntimeError:
        return {
            "fit_success": False,
            "omega_fit_meV": float("nan"),
            "gamma_fit_meV": float("nan"),
            "amp_fit": float("nan"),
            "elastic_fit": float("nan"),
            "bg_fit": float("nan"),
            "fit_r2": float("nan"),
            "fit_mae": float("nan"),
        }

    omega_fit, gamma_fit, amp_fit, elastic_fit, bg_fit = [float(v) for v in popt]
    y_fit = model(omega_meV, omega_fit, gamma_fit, amp_fit, elastic_fit, bg_fit)
    r2 = float(r2_score(counts, y_fit))
    mae = float(mean_absolute_error(counts, y_fit))

    return {
        "fit_success": True,
        "omega_fit_meV": omega_fit,
        "gamma_fit_meV": gamma_fit,
        "amp_fit": amp_fit,
        "elastic_fit": elastic_fit,
        "bg_fit": bg_fit,
        "fit_r2": r2,
        "fit_mae": mae,
    }


def fit_ixs_map(q: np.ndarray, omega: np.ndarray, observed: np.ndarray, params: IXSParams) -> pd.DataFrame:
    """Run per-Q spectrum fitting and collect a summary dataframe."""

    rows: list[dict[str, float | bool]] = []

    for iq, qv in enumerate(q):
        fit_res = fit_single_q_spectrum(
            omega_meV=omega,
            counts=observed[iq, :],
            temperature_K=params.temperature_K,
            resolution_sigma_meV=params.resolution_sigma_meV,
        )

        true_omega = float(phonon_dispersion(qv, params.sound_velocity_true_meV_A, params.gap_true_meV))
        true_gamma = float(phonon_damping(qv, params.gamma0_true_meV, params.gamma2_true_meV_A2))

        row = {
            "q_Ainv": float(qv),
            "omega_true_meV": true_omega,
            "gamma_true_meV": true_gamma,
        }
        row.update(fit_res)
        rows.append(row)

    return pd.DataFrame(rows)


def regress_dispersion(fit_df: pd.DataFrame) -> dict[str, float]:
    """Fit Omega(Q)^2 = c^2 Q^2 + gap^2 with linear regression."""

    valid = fit_df[fit_df["fit_success"]].copy()
    if len(valid) < 8:
        raise RuntimeError("too few successful Q-fits for dispersion regression")

    x = (valid["q_Ainv"].to_numpy() ** 2).reshape(-1, 1)
    y = valid["omega_fit_meV"].to_numpy() ** 2

    reg = LinearRegression()
    reg.fit(x, y)

    coef = float(reg.coef_[0])
    intercept = float(reg.intercept_)

    c_est = float(np.sqrt(max(coef, 0.0)))
    gap_est = float(np.sqrt(max(intercept, 0.0)))

    mae = float(mean_absolute_error(valid["omega_true_meV"], valid["omega_fit_meV"]))
    r2 = float(reg.score(x, y))

    return {
        "c_est_meV_A": c_est,
        "gap_est_meV": gap_est,
        "dispersion_r2": r2,
        "omega_fit_mae_meV": mae,
        "n_reg_points": float(len(valid)),
    }


def _inv_softplus(x: float) -> float:
    x_clamped = max(float(x), 1e-8)
    return float(np.log(np.expm1(x_clamped)))


def _torch_lorentzian(w: torch.Tensor, center: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return gamma / np.pi / ((w - center) ** 2 + gamma**2)


def _gaussian_kernel1d_torch(sigma_px: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    radius = max(1, int(np.ceil(4.0 * sigma_px)))
    grid = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (grid / sigma_px) ** 2)
    kernel = kernel / torch.sum(kernel)
    return kernel.view(1, 1, -1)


def torch_global_refine(
    q: np.ndarray,
    omega: np.ndarray,
    observed: np.ndarray,
    params: IXSParams,
) -> dict[str, float]:
    """Use PyTorch to globally refine dispersion and damping parameters."""

    torch.manual_seed(params.random_seed)

    q_ds = q[::2]
    omega_ds = omega[::2]
    obs_ds = observed[::2, ::2]

    scale = float(np.max(obs_ds))
    obs_norm = obs_ds / max(scale, 1e-8)

    dtype = torch.float64
    device = torch.device("cpu")

    q_t = torch.tensor(q_ds, dtype=dtype, device=device)
    w_t = torch.tensor(omega_ds, dtype=dtype, device=device)
    obs_t = torch.tensor(obs_norm, dtype=dtype, device=device)

    sigma_px = params.resolution_sigma_meV / float(omega_ds[1] - omega_ds[0])
    kernel = _gaussian_kernel1d_torch(sigma_px=sigma_px, dtype=dtype, device=device)

    raw_c = torch.nn.Parameter(torch.tensor(_inv_softplus(8.5), dtype=dtype, device=device))
    raw_gap = torch.nn.Parameter(torch.tensor(_inv_softplus(6.5), dtype=dtype, device=device))
    raw_g0 = torch.nn.Parameter(torch.tensor(_inv_softplus(1.8), dtype=dtype, device=device))
    raw_g2 = torch.nn.Parameter(torch.tensor(_inv_softplus(0.2), dtype=dtype, device=device))
    raw_amp0 = torch.nn.Parameter(torch.tensor(_inv_softplus(700.0), dtype=dtype, device=device))
    raw_el0 = torch.nn.Parameter(torch.tensor(_inv_softplus(120.0), dtype=dtype, device=device))
    raw_el_decay = torch.nn.Parameter(torch.tensor(_inv_softplus(2.2), dtype=dtype, device=device))
    raw_bg = torch.nn.Parameter(torch.tensor(_inv_softplus(1.0), dtype=dtype, device=device))

    optimizer = torch.optim.Adam(
        [raw_c, raw_gap, raw_g0, raw_g2, raw_amp0, raw_el0, raw_el_decay, raw_bg],
        lr=params.torch_lr,
        weight_decay=params.torch_weight_decay,
    )

    kb_t = KB_MEV_PER_K * params.temperature_K

    for _ in range(params.torch_epochs):
        optimizer.zero_grad()

        c = torch.nn.functional.softplus(raw_c)
        gap = torch.nn.functional.softplus(raw_gap)
        g0 = torch.nn.functional.softplus(raw_g0)
        g2 = torch.nn.functional.softplus(raw_g2)
        amp0 = torch.nn.functional.softplus(raw_amp0)
        el0 = torch.nn.functional.softplus(raw_el0)
        el_decay = torch.nn.functional.softplus(raw_el_decay)
        bg = torch.nn.functional.softplus(raw_bg)

        qv = q_t[:, None]
        wv = w_t[None, :]

        omega0 = torch.sqrt((c * qv) ** 2 + gap**2)
        gamma = g0 + g2 * qv**2

        amp_q = amp0 * qv**2 / (1.0 + (qv / 2.0) ** 2)
        elastic_q = el0 * torch.exp(-qv / el_decay)

        x_bose = torch.clamp(omega0 / kb_t, min=1e-8, max=80.0)
        n_bose = 1.0 / (torch.exp(x_bose) - 1.0)

        inel = amp_q * (
            (n_bose + 1.0) * _torch_lorentzian(wv, omega0, gamma)
            + n_bose * _torch_lorentzian(wv, -omega0, gamma)
        )
        elastic = elastic_q * _torch_lorentzian(wv, torch.zeros_like(wv), torch.full_like(wv, 0.2))

        raw_map = inel + elastic + bg
        conv_map = torch.nn.functional.conv1d(
            raw_map.unsqueeze(1),
            kernel,
            padding=kernel.shape[-1] // 2,
        ).squeeze(1)

        pred_norm = conv_map / max(scale, 1e-8)
        loss = torch.mean((pred_norm - obs_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        c_fit = float(torch.nn.functional.softplus(raw_c).cpu().item())
        gap_fit = float(torch.nn.functional.softplus(raw_gap).cpu().item())
        g0_fit = float(torch.nn.functional.softplus(raw_g0).cpu().item())
        g2_fit = float(torch.nn.functional.softplus(raw_g2).cpu().item())
        final_mse = float(loss.cpu().item())

    return {
        "torch_c_fit_meV_A": c_fit,
        "torch_gap_fit_meV": gap_fit,
        "torch_g0_fit_meV": g0_fit,
        "torch_g2_fit_meV_A2": g2_fit,
        "torch_final_mse": final_mse,
    }


def main() -> None:
    params = IXSParams()
    check_params(params)

    q, omega, _expected, observed = simulate_ixs_map(params)
    fit_df = fit_ixs_map(q=q, omega=omega, observed=observed, params=params)

    successful = fit_df[fit_df["fit_success"]].copy()
    n_success = int(len(successful))
    mean_r2 = float(successful["fit_r2"].mean())

    dispersion = regress_dispersion(fit_df)
    torch_res = torch_global_refine(q=q, omega=omega, observed=observed, params=params)

    summary = {
        "n_q_total": int(params.n_q),
        "n_q_success": n_success,
        "mean_fit_r2": mean_r2,
        "dispersion_r2": float(dispersion["dispersion_r2"]),
        "omega_fit_mae_meV": float(dispersion["omega_fit_mae_meV"]),
        "regressed_c_meV_A": float(dispersion["c_est_meV_A"]),
        "regressed_gap_meV": float(dispersion["gap_est_meV"]),
        "torch_c_fit_meV_A": float(torch_res["torch_c_fit_meV_A"]),
        "torch_gap_fit_meV": float(torch_res["torch_gap_fit_meV"]),
        "torch_g0_fit_meV": float(torch_res["torch_g0_fit_meV"]),
        "torch_g2_fit_meV_A2": float(torch_res["torch_g2_fit_meV_A2"]),
        "torch_final_mse": float(torch_res["torch_final_mse"]),
        "true_c_meV_A": params.sound_velocity_true_meV_A,
        "true_gap_meV": params.gap_true_meV,
        "true_g0_meV": params.gamma0_true_meV,
        "true_g2_meV_A2": params.gamma2_true_meV_A2,
    }

    print("=== IXS MVP Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    print("\n=== Per-Q Fit (head) ===")
    display_cols = [
        "q_Ainv",
        "omega_true_meV",
        "omega_fit_meV",
        "gamma_true_meV",
        "gamma_fit_meV",
        "fit_r2",
        "fit_mae",
        "fit_success",
    ]
    print(fit_df[display_cols].head(10).to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    # Quality gates for this MVP pipeline.
    if n_success < int(0.75 * params.n_q):
        raise AssertionError("Too many failed Q-slice fits")
    if mean_r2 < 0.90:
        raise AssertionError("Per-Q fit quality is too low")
    if dispersion["dispersion_r2"] < 0.94:
        raise AssertionError("Dispersion regression quality is too low")
    if dispersion["omega_fit_mae_meV"] > 2.5:
        raise AssertionError("Recovered phonon energy MAE is too large")
    if abs(torch_res["torch_c_fit_meV_A"] - params.sound_velocity_true_meV_A) > 3.0:
        raise AssertionError("Torch-estimated sound velocity deviates too much")
    if torch_res["torch_final_mse"] > 4.0e-3:
        raise AssertionError("Torch global fitting MSE is too high")


if __name__ == "__main__":
    main()
