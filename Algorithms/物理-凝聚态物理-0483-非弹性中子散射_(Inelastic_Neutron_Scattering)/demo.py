"""MVP: Inelastic Neutron Scattering (INS) workflow on synthetic data.

This script demonstrates a compact but honest algorithmic pipeline:
1) Simulate a dynamic structure factor map S(Q, omega).
2) Denoise and detect peak positions at each Q.
3) Refine peak centers with local Lorentzian fits.
4) Fit a global dispersion model with SciPy and PyTorch.
5) Export tabular results and print a concise report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class TrueParams:
    delta: float = 2.0
    velocity: float = 8.0
    gamma: float = 0.55


def dispersion_model(q: np.ndarray, delta: float, velocity: float) -> np.ndarray:
    """Gapped 1D-like mode: omega(q) = sqrt(delta^2 + (v*sin(pi*q))^2)."""
    return np.sqrt(delta**2 + (velocity * np.sin(np.pi * q)) ** 2)


def lorentzian(omega: np.ndarray, amplitude: float, center: float, gamma: float, bg: float) -> np.ndarray:
    """Lorentzian line shape with constant background."""
    return amplitude * (gamma**2 / ((omega - center) ** 2 + gamma**2)) + bg


def simulate_ins_map(
    q_grid: np.ndarray,
    omega_grid: np.ndarray,
    params: TrueParams,
    noise_std: float = 0.08,
    seed: int = 7,
) -> np.ndarray:
    """Create synthetic S(Q, omega) with one dispersive branch and weak background."""
    rng = np.random.default_rng(seed)
    omega0 = dispersion_model(q_grid, params.delta, params.velocity)

    s_map = np.zeros((q_grid.size, omega_grid.size), dtype=np.float64)
    for i, (qv, w0) in enumerate(zip(q_grid, omega0)):
        amplitude = 1.4 * np.exp(-0.8 * (qv - 0.5) ** 2) + 0.6
        bg = 0.06 + 0.02 * np.cos(2.0 * np.pi * qv)
        line = lorentzian(omega_grid, amplitude=amplitude, center=w0, gamma=params.gamma, bg=bg)

        # Add a weak nondispersive feature to make peak extraction non-trivial.
        weak_mode = 0.12 * np.exp(-((omega_grid - 5.5) ** 2) / (2.0 * 0.45**2))
        noise = rng.normal(0.0, noise_std, size=omega_grid.size)
        s_map[i, :] = np.clip(line + weak_mode + noise, a_min=0.0, a_max=None)

    return s_map


def preprocess_map(s_map: np.ndarray) -> np.ndarray:
    """Scale + smooth each Q-slice for robust peak detection."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(s_map)
    smoothed = gaussian_filter1d(scaled, sigma=1.2, axis=1)
    return smoothed


def extract_peak_centers(
    q_grid: np.ndarray,
    omega_grid: np.ndarray,
    s_proc: np.ndarray,
) -> pd.DataFrame:
    """Detect dominant peak in each Q-slice and refine with local Lorentzian fit."""
    rows: list[dict[str, float]] = []
    wmin, wmax = float(omega_grid.min()), float(omega_grid.max())

    for i, qv in enumerate(q_grid):
        spectrum = s_proc[i]
        peaks, props = find_peaks(spectrum, prominence=0.25)
        if peaks.size == 0:
            continue

        best_idx = int(peaks[np.argmax(props["prominences"])])
        w_guess = float(omega_grid[best_idx])

        # Local fitting window around the detected peak.
        mask = np.abs(omega_grid - w_guess) <= 2.0
        x = omega_grid[mask]
        y = spectrum[mask]
        if x.size < 8:
            continue

        p0 = [float(y.max() - y.min()), w_guess, 0.45, float(np.median(y))]
        bounds = ([0.0, wmin, 0.05, -3.0], [20.0, wmax, 3.0, 3.0])

        try:
            popt, _ = curve_fit(lorentzian, x, y, p0=p0, bounds=bounds, maxfev=5000)
            amp, center, gamma, bg = map(float, popt)
        except RuntimeError:
            amp, center, gamma, bg = float("nan"), w_guess, float("nan"), float("nan")

        rows.append(
            {
                "Q": float(qv),
                "omega_peak": center,
                "amplitude": amp,
                "gamma": gamma,
                "background": bg,
                "omega_guess": w_guess,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["omega_peak", "gamma"])
    df = df[(df["omega_peak"] > 0.0) & (df["gamma"] > 0.0)]
    return df.sort_values("Q").reset_index(drop=True)


def fit_dispersion_scipy(df_peaks: pd.DataFrame) -> tuple[float, float]:
    """Fit (delta, velocity) with nonlinear least squares."""
    q = df_peaks["Q"].to_numpy()
    w = df_peaks["omega_peak"].to_numpy()

    p0 = [1.5, 7.0]
    bounds = ([0.1, 0.1], [10.0, 20.0])
    popt, _ = curve_fit(dispersion_model, q, w, p0=p0, bounds=bounds, maxfev=8000)
    return float(popt[0]), float(popt[1])


def fit_dispersion_torch(df_peaks: pd.DataFrame, steps: int = 1200, lr: float = 0.03) -> tuple[float, float]:
    """Fit the same dispersion model with PyTorch autodiff (Huber loss)."""
    q = torch.tensor(df_peaks["Q"].to_numpy(), dtype=torch.float32)
    w = torch.tensor(df_peaks["omega_peak"].to_numpy(), dtype=torch.float32)

    raw_delta = torch.nn.Parameter(torch.tensor(1.0))
    raw_velocity = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam([raw_delta, raw_velocity], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        delta = torch.nn.functional.softplus(raw_delta) + 1e-4
        velocity = torch.nn.functional.softplus(raw_velocity) + 1e-4
        pred = torch.sqrt(delta**2 + (velocity * torch.sin(torch.pi * q)) ** 2)
        loss = torch.nn.functional.huber_loss(pred, w, delta=0.2)
        loss.backward()
        optimizer.step()

    delta_out = float(torch.nn.functional.softplus(raw_delta).detach().cpu().numpy())
    velocity_out = float(torch.nn.functional.softplus(raw_velocity).detach().cpu().numpy())
    return delta_out, velocity_out


def summarize_fit(df_peaks: pd.DataFrame, delta: float, velocity: float) -> dict[str, float]:
    q = df_peaks["Q"].to_numpy()
    w_true = df_peaks["omega_peak"].to_numpy()
    w_pred = dispersion_model(q, delta, velocity)
    return {
        "mae": float(mean_absolute_error(w_true, w_pred)),
        "r2": float(r2_score(w_true, w_pred)),
        "n_points": int(w_true.size),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    # Synthetic measurement grid
    q_grid = np.linspace(0.05, 0.95, 70)
    omega_grid = np.linspace(0.2, 12.0, 260)
    true_params = TrueParams(delta=2.0, velocity=8.0, gamma=0.55)

    s_map = simulate_ins_map(q_grid, omega_grid, params=true_params)
    s_proc = preprocess_map(s_map)
    df_peaks = extract_peak_centers(q_grid, omega_grid, s_proc)

    if df_peaks.empty or len(df_peaks) < 20:
        raise RuntimeError("Peak extraction failed: not enough points for dispersion fitting.")

    delta_sp, velocity_sp = fit_dispersion_scipy(df_peaks)
    delta_torch, velocity_torch = fit_dispersion_torch(df_peaks)

    metrics_sp = summarize_fit(df_peaks, delta_sp, velocity_sp)
    metrics_torch = summarize_fit(df_peaks, delta_torch, velocity_torch)

    # Save artifacts for reproducibility.
    pd.DataFrame(
        {
            "Q": q_grid,
            **{f"S_w{j}": s_map[:, j] for j in range(min(16, s_map.shape[1]))},
        }
    ).to_csv(out_dir / "ins_map_preview.csv", index=False)
    df_peaks.to_csv(out_dir / "peak_table.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "method": "scipy_curve_fit",
                "delta": delta_sp,
                "velocity": velocity_sp,
                **metrics_sp,
            },
            {
                "method": "pytorch_autodiff",
                "delta": delta_torch,
                "velocity": velocity_torch,
                **metrics_torch,
            },
            {
                "method": "ground_truth",
                "delta": true_params.delta,
                "velocity": true_params.velocity,
                "mae": np.nan,
                "r2": np.nan,
                "n_points": metrics_sp["n_points"],
            },
        ]
    )
    summary.to_csv(out_dir / "fit_summary.csv", index=False)

    print("INS MVP finished.")
    print(f"Detected peak points: {metrics_sp['n_points']}")
    print(
        "SciPy fit: "
        f"delta={delta_sp:.3f}, v={velocity_sp:.3f}, "
        f"MAE={metrics_sp['mae']:.4f}, R2={metrics_sp['r2']:.4f}"
    )
    print(
        "PyTorch fit: "
        f"delta={delta_torch:.3f}, v={velocity_torch:.3f}, "
        f"MAE={metrics_torch['mae']:.4f}, R2={metrics_torch['r2']:.4f}"
    )
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
