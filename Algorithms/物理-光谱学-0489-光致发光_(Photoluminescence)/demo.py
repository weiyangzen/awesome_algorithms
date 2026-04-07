"""Minimal runnable MVP for photoluminescence concentration estimation.

Pipeline:
1) Synthesize PL spectra with target/interferent emission, quenching, inner-filter effect,
   baseline drift, wavelength warping, and noise.
2) Preprocess spectra (Savitzky-Golay smoothing + low-quantile polynomial baseline correction).
3) Estimate target concentration by:
   - Single-wavelength linear calibration (interpretable baseline)
   - Full-spectrum Ridge regression
   - Full-spectrum PyTorch MLP
4) Evaluate on in-distribution and shifted test sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PLDataset:
    """Container for simulated photoluminescence data."""

    spectra: np.ndarray
    c_target: np.ndarray
    c_interferent: np.ndarray
    quencher: np.ndarray


class SmallMLP(nn.Module):
    """Small nonlinear regressor for PL concentration estimation."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gaussian_peak(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Return a Gaussian line shape."""
    z = (x - center) / sigma
    return np.exp(-0.5 * z * z)


def build_emission_profiles(wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic target/interferent PL emission profiles."""
    target = (
        1.20 * gaussian_peak(wavelengths, 548.0, 17.0)
        + 0.70 * gaussian_peak(wavelengths, 585.0, 24.0)
    )
    interferent = (
        0.85 * gaussian_peak(wavelengths, 515.0, 20.0)
        + 0.95 * gaussian_peak(wavelengths, 620.0, 32.0)
        + 0.25 * gaussian_peak(wavelengths, 700.0, 42.0)
    )
    return target, interferent


def simulate_pl_dataset(
    n_samples: int,
    wavelengths: np.ndarray,
    target_profile: np.ndarray,
    interferent_profile: np.ndarray,
    seed: int,
    shifted: bool,
) -> PLDataset:
    """Generate PL spectra under realistic perturbations.

    shifted=True introduces stronger quenching, higher interferent levels, and larger noise.
    """
    rng = np.random.default_rng(seed)
    n_wl = wavelengths.size

    c_target = rng.uniform(0.05, 1.20, size=n_samples)
    if shifted:
        c_interferent = rng.uniform(0.18, 1.05, size=n_samples)
        quencher = rng.uniform(0.20, 0.95, size=n_samples)
        noise_sigma = 0.020
        beta_inner = 0.26
        k_sv = 1.65
        shift_sigma_nm = 1.6
    else:
        c_interferent = rng.uniform(0.00, 0.70, size=n_samples)
        quencher = rng.uniform(0.00, 0.65, size=n_samples)
        noise_sigma = 0.011
        beta_inner = 0.18
        k_sv = 1.10
        shift_sigma_nm = 0.8

    gain = rng.uniform(0.90, 1.10, size=n_samples)

    amp_target = (
        gain
        * c_target
        * np.exp(-beta_inner * (c_target + c_interferent))
        / (1.0 + k_sv * quencher)
    )
    amp_interferent = gain * c_interferent * (1.0 - 0.15 * quencher)

    core = np.outer(amp_target, target_profile) + np.outer(amp_interferent, interferent_profile)

    shifted_core = np.empty_like(core)
    for i in range(n_samples):
        delta = rng.normal(0.0, shift_sigma_nm)
        shifted_core[i] = np.interp(
            wavelengths,
            wavelengths + delta,
            core[i],
            left=core[i, 0],
            right=core[i, -1],
        )

    x = (wavelengths - wavelengths.mean()) / (0.5 * (wavelengths.max() - wavelengths.min()))
    b0 = rng.normal(0.0, 0.010 if shifted else 0.006, size=(n_samples, 1))
    b1 = rng.normal(0.0, 0.035 if shifted else 0.020, size=(n_samples, 1))
    b2 = rng.normal(0.0, 0.018 if shifted else 0.010, size=(n_samples, 1))
    baseline = b0 + b1 * x[None, :] + b2 * (x[None, :] ** 2)

    shot_sigma = 0.030 if shifted else 0.020
    noise = rng.normal(0.0, noise_sigma, size=(n_samples, n_wl))
    noise += rng.normal(0.0, shot_sigma * np.sqrt(np.clip(shifted_core, 0.0, None)))

    spectra = shifted_core + baseline + noise
    return PLDataset(
        spectra=spectra,
        c_target=c_target,
        c_interferent=c_interferent,
        quencher=quencher,
    )


def preprocess_spectra(spectra: np.ndarray) -> np.ndarray:
    """Smooth spectra and remove baseline using low-quantile polynomial fitting."""
    smoothed = savgol_filter(spectra, window_length=9, polyorder=3, axis=1, mode="interp")

    n_samples, n_wl = smoothed.shape
    x = np.linspace(-1.0, 1.0, n_wl)
    corrected = np.empty_like(smoothed)

    for i in range(n_samples):
        row = smoothed[i]
        threshold = np.quantile(row, 0.32)
        mask = row <= threshold

        if np.count_nonzero(mask) < 14:
            idx = np.argsort(row)[:14]
            mask = np.zeros_like(row, dtype=bool)
            mask[idx] = True

        coeff = np.polyfit(x[mask], row[mask], deg=2)
        baseline = np.polyval(coeff, x)
        corrected[i] = row - baseline

    return corrected


def select_analytical_wavelength(
    wavelengths: np.ndarray,
    target_profile: np.ndarray,
    interferent_profile: np.ndarray,
) -> tuple[int, float]:
    """Choose target peak with high target/interferent separation."""
    peaks, _ = find_peaks(
        target_profile,
        prominence=0.10 * float(np.max(target_profile)),
        distance=12,
    )

    if peaks.size == 0:
        idx = int(np.argmax(target_profile))
        return idx, float(wavelengths[idx])

    denominator = interferent_profile[peaks] + 0.05 * float(np.max(interferent_profile)) + 1e-12
    score = target_profile[peaks] / denominator
    best = int(peaks[int(np.argmax(score))])
    return best, float(wavelengths[best])


def fit_single_wavelength_calibration(x_peak: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit linear calibration c = slope * I_peak + intercept."""
    slope, intercept = np.polyfit(x_peak, y, deg=1)
    return float(slope), float(intercept)


def predict_single_wavelength(x_peak: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Predict concentration from single-wavelength intensity."""
    pred = slope * x_peak + intercept
    return np.clip(pred, 0.0, None)


def train_ridge_model(x_train: np.ndarray, y_train: np.ndarray) -> object:
    """Train full-spectrum ridge regressor."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    return model


def train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
    epochs: int = 110,
    lr: float = 8e-4,
) -> tuple[SmallMLP, StandardScaler, float, float, float]:
    """Train an MLP on standardized spectra and normalized targets."""
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(x_train)
    x_train_std = scaler.transform(x_train).astype(np.float32)
    x_val_std = scaler.transform(x_val).astype(np.float32)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_norm = ((y_train - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    y_val_norm = ((y_val - y_mean) / y_std).astype(np.float32).reshape(-1, 1)

    dataset = TensorDataset(torch.from_numpy(x_train_std), torch.from_numpy(y_train_norm))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SmallMLP(in_dim=x_train_std.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_val_tensor = torch.from_numpy(x_val_std)
    y_val_tensor = torch.from_numpy(y_val_norm)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_tensor)
            val_loss = float(loss_fn(val_pred, y_val_tensor).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, y_mean, y_std, best_val


def predict_torch_mlp(
    model: SmallMLP,
    scaler: StandardScaler,
    y_mean: float,
    y_std: float,
    x: np.ndarray,
) -> np.ndarray:
    """Predict concentration with trained MLP and inverse target normalization."""
    model.eval()
    x_std = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_std)).squeeze(1).cpu().numpy()
    pred = pred_norm * y_std + y_mean
    return np.clip(pred, 0.0, None)


def metrics_row(model_name: str, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute common regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    wavelengths = np.arange(420.0, 781.0, 1.0)
    target_profile, interferent_profile = build_emission_profiles(wavelengths)

    train_data = simulate_pl_dataset(
        n_samples=420,
        wavelengths=wavelengths,
        target_profile=target_profile,
        interferent_profile=interferent_profile,
        seed=1,
        shifted=False,
    )
    test_id = simulate_pl_dataset(
        n_samples=180,
        wavelengths=wavelengths,
        target_profile=target_profile,
        interferent_profile=interferent_profile,
        seed=2,
        shifted=False,
    )
    test_shifted = simulate_pl_dataset(
        n_samples=180,
        wavelengths=wavelengths,
        target_profile=target_profile,
        interferent_profile=interferent_profile,
        seed=3,
        shifted=True,
    )

    x_train = preprocess_spectra(train_data.spectra)
    x_test_id = preprocess_spectra(test_id.spectra)
    x_test_shifted = preprocess_spectra(test_shifted.spectra)
    y_train = train_data.c_target

    peak_idx, peak_wavelength = select_analytical_wavelength(
        wavelengths=wavelengths,
        target_profile=target_profile,
        interferent_profile=interferent_profile,
    )

    slope, intercept = fit_single_wavelength_calibration(x_train[:, peak_idx], y_train)

    ridge = train_ridge_model(x_train, y_train)

    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=7)
    mlp, mlp_scaler, y_mean, y_std, best_val = train_torch_mlp(
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_val,
        y_val=y_val,
        seed=7,
        epochs=110,
        lr=8e-4,
    )

    pred_single_id = predict_single_wavelength(x_test_id[:, peak_idx], slope, intercept)
    pred_single_shift = predict_single_wavelength(x_test_shifted[:, peak_idx], slope, intercept)

    pred_ridge_id = np.clip(ridge.predict(x_test_id), 0.0, None)
    pred_ridge_shift = np.clip(ridge.predict(x_test_shifted), 0.0, None)

    pred_mlp_id = predict_torch_mlp(mlp, mlp_scaler, y_mean, y_std, x_test_id)
    pred_mlp_shift = predict_torch_mlp(mlp, mlp_scaler, y_mean, y_std, x_test_shifted)

    rows = [
        metrics_row("SinglePeak", "ID", test_id.c_target, pred_single_id),
        metrics_row("SinglePeak", "Shifted", test_shifted.c_target, pred_single_shift),
        metrics_row("Ridge", "ID", test_id.c_target, pred_ridge_id),
        metrics_row("Ridge", "Shifted", test_shifted.c_target, pred_ridge_shift),
        metrics_row("TorchMLP", "ID", test_id.c_target, pred_mlp_id),
        metrics_row("TorchMLP", "Shifted", test_shifted.c_target, pred_mlp_shift),
    ]
    metrics_df = pd.DataFrame(rows)

    print("=== Photoluminescence MVP ===")
    print(f"Wavelength grid: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm ({wavelengths.size} points)")
    print(f"Selected analytical wavelength: {peak_wavelength:.1f} nm (index={peak_idx})")
    print(f"Single-peak calibration: c = {slope:.4f} * I_peak + {intercept:.4f}")
    print(f"TorchMLP best validation MSE (normalized target): {best_val:.6f}")
    print()

    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sample_df = pd.DataFrame(
        {
            "True_c": test_shifted.c_target[:8],
            "SinglePeak": pred_single_shift[:8],
            "Ridge": pred_ridge_shift[:8],
            "TorchMLP": pred_mlp_shift[:8],
        }
    )
    print("\nShifted set sample predictions (first 8):")
    print(sample_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
