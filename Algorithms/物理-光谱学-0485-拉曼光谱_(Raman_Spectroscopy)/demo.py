"""Minimal runnable MVP for Raman spectroscopy concentration estimation.

Pipeline:
1) Synthesize Raman spectra with analyte/interferent peaks, fluorescence baseline, noise, and spikes.
2) Preprocess each spectrum (despike + AsLS baseline correction + Savitzky-Golay smoothing).
3) Quantify analyte concentration by three approaches:
   - Single-peak linear calibration (interpretable baseline)
   - Full-spectrum Ridge regression
   - Full-spectrum PyTorch MLP
4) Evaluate on in-distribution and shifted test sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RamanDataset:
    """Container for simulated Raman spectra and concentrations."""

    spectra: np.ndarray
    c_target: np.ndarray
    c_interferent: np.ndarray


class SmallMLP(nn.Module):
    """Small nonlinear regressor for Raman concentration prediction."""

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


def gaussian_peak(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return Gaussian peak profile."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z)


def build_reference_profiles(shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build analyte/interferent reference Raman profiles."""
    target = (
        0.95 * gaussian_peak(shifts, 620.0, 15.0)
        + 1.35 * gaussian_peak(shifts, 1002.0, 12.0)
        + 0.90 * gaussian_peak(shifts, 1605.0, 21.0)
    )
    interferent = (
        0.80 * gaussian_peak(shifts, 724.0, 18.0)
        + 1.05 * gaussian_peak(shifts, 1034.0, 14.0)
        + 0.72 * gaussian_peak(shifts, 1450.0, 28.0)
        + 0.32 * gaussian_peak(shifts, 1710.0, 36.0)
    )
    return target, interferent


def simulate_raman_dataset(
    n_samples: int,
    shifts: np.ndarray,
    target_ref: np.ndarray,
    interferent_ref: np.ndarray,
    seed: int,
    shifted: bool,
) -> RamanDataset:
    """Generate Raman spectra with baseline, noise, shift, and occasional spikes."""
    rng = np.random.default_rng(seed)
    n_shift = shifts.size

    c_target = rng.uniform(0.05, 1.20, size=n_samples)
    if shifted:
        c_interferent = rng.uniform(0.18, 1.10, size=n_samples)
        noise_sigma = 0.020
        shift_sigma = 2.6
        fluorescence_amp = (0.12, 0.35)
        baseline_scale = 1.35
    else:
        c_interferent = rng.uniform(0.00, 0.75, size=n_samples)
        noise_sigma = 0.012
        shift_sigma = 1.2
        fluorescence_amp = (0.05, 0.22)
        baseline_scale = 1.0

    x_norm = (shifts - shifts.mean()) / (0.5 * (shifts.max() - shifts.min()))

    spectra = np.empty((n_samples, n_shift), dtype=np.float64)
    for i in range(n_samples):
        core = c_target[i] * target_ref + c_interferent[i] * interferent_ref

        delta = rng.normal(0.0, shift_sigma)
        warped = np.interp(shifts, shifts + delta, core, left=core[0], right=core[-1])

        b0 = rng.normal(0.0, 0.02 * baseline_scale)
        b1 = rng.normal(0.0, 0.07 * baseline_scale)
        b2 = rng.normal(0.0, 0.05 * baseline_scale)
        polynomial_baseline = b0 + b1 * x_norm + b2 * (x_norm**2)

        f_amp = rng.uniform(*fluorescence_amp)
        fluorescence = f_amp * np.exp(-(shifts - shifts.min()) / 580.0)

        noise = rng.normal(0.0, noise_sigma, size=n_shift)

        spec = warped + polynomial_baseline + fluorescence + noise

        n_spikes = int(rng.integers(0, 3 if shifted else 2))
        if n_spikes > 0:
            idx = rng.choice(n_shift, size=n_spikes, replace=False)
            spec[idx] += rng.uniform(0.45, 1.20, size=n_spikes)

        spectra[i] = spec

    return RamanDataset(spectra=spectra, c_target=c_target, c_interferent=c_interferent)


def despike_spectrum(y: np.ndarray, kernel_size: int = 5, threshold: float = 7.0) -> np.ndarray:
    """Remove cosmic spikes using median filtering + MAD thresholding."""
    smooth_med = medfilt(y, kernel_size=kernel_size)
    residual = y - smooth_med
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-12
    robust_sigma = 1.4826 * mad
    mask = np.abs(residual) > threshold * robust_sigma

    out = y.copy()
    out[mask] = smooth_med[mask]
    return out


def asls_baseline(
    y: np.ndarray,
    dtd: sparse.csc_matrix,
    lam: float = 2e5,
    p: float = 0.01,
    n_iter: int = 7,
) -> np.ndarray:
    """Asymmetric least squares baseline estimation.

    Solves iteratively:
    min_z sum_i w_i (y_i - z_i)^2 + lam * ||D2 z||^2
    """
    n = y.size
    w = np.ones(n, dtype=np.float64)

    for _ in range(n_iter):
        w_mat = sparse.spdiags(w, 0, n, n, format="csc")
        z = spsolve(w_mat + lam * dtd, w * y)
        w = np.where(y > z, p, 1.0 - p)

    return np.asarray(z)


def preprocess_spectra(
    spectra: np.ndarray,
    asls_lambda: float = 2e5,
    asls_p: float = 0.01,
    asls_iter: int = 7,
) -> np.ndarray:
    """Apply despike + AsLS baseline correction + Savitzky-Golay smoothing."""
    n_samples, n_points = spectra.shape
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_points - 2, n_points), format="csc")
    dtd = (d2.T @ d2).tocsc()

    out = np.empty_like(spectra)
    for i in range(n_samples):
        y = despike_spectrum(spectra[i])
        baseline = asls_baseline(y, dtd=dtd, lam=asls_lambda, p=asls_p, n_iter=asls_iter)
        corrected = y - baseline
        out[i] = savgol_filter(corrected, window_length=11, polyorder=3, mode="interp")

    return out


def select_analytical_peak(
    shifts: np.ndarray, target_ref: np.ndarray, interferent_ref: np.ndarray
) -> tuple[int, float, float]:
    """Select analyte peak with high target/interferent separation."""
    peaks, _ = find_peaks(target_ref, prominence=0.08 * float(np.max(target_ref)), distance=10)

    if peaks.size == 0:
        idx = int(np.argmax(target_ref))
        denom = float(interferent_ref[idx]) + 1e-12
        return idx, float(shifts[idx]), float(target_ref[idx] / denom)

    denom = interferent_ref[peaks] + 0.05 * float(np.max(interferent_ref)) + 1e-12
    separation = target_ref[peaks] / denom
    best_local = int(np.argmax(separation))
    best_idx = int(peaks[best_local])
    return best_idx, float(shifts[best_idx]), float(separation[best_local])


def fit_single_peak_calibration(x_peak: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = slope * x_peak + intercept."""
    slope, intercept = np.polyfit(x_peak, y, deg=1)
    return float(slope), float(intercept)


def predict_single_peak(x_peak: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Predict concentration from one Raman peak intensity."""
    pred = slope * x_peak + intercept
    return np.clip(pred, 0.0, None)


def train_ridge_model(x_train: np.ndarray, y_train: np.ndarray) -> object:
    """Train full-spectrum scaled Ridge model."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    return model


def train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
    epochs: int = 120,
    lr: float = 8e-4,
) -> tuple[SmallMLP, StandardScaler, float, float, float]:
    """Train small MLP on standardized spectra."""
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(x_train)
    x_train_std = scaler.transform(x_train).astype(np.float32)
    x_val_std = scaler.transform(x_val).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_f = ((y_train - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    y_val_f = ((y_val - y_mean) / y_std).astype(np.float32).reshape(-1, 1)

    train_dataset = TensorDataset(torch.from_numpy(x_train_std), torch.from_numpy(y_train_f))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SmallMLP(in_dim=x_train_std.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_val_tensor = torch.from_numpy(x_val_std)
    y_val_tensor = torch.from_numpy(y_val_f)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
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
    """Predict concentration using trained MLP."""
    model.eval()
    x_std = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_std)).squeeze(1).cpu().numpy()
    pred = pred_norm * y_std + y_mean
    return np.clip(pred, 0.0, None)


def metrics_row(model: str, dataset: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    """Compute regression metrics row."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model,
        "Dataset": dataset,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def main() -> None:
    np.random.seed(13)
    torch.manual_seed(13)

    shifts = np.arange(200.0, 2000.0, 2.0)
    target_ref, interferent_ref = build_reference_profiles(shifts)

    train_set = simulate_raman_dataset(
        n_samples=240,
        shifts=shifts,
        target_ref=target_ref,
        interferent_ref=interferent_ref,
        seed=101,
        shifted=False,
    )
    test_id = simulate_raman_dataset(
        n_samples=80,
        shifts=shifts,
        target_ref=target_ref,
        interferent_ref=interferent_ref,
        seed=202,
        shifted=False,
    )
    test_shift = simulate_raman_dataset(
        n_samples=80,
        shifts=shifts,
        target_ref=target_ref,
        interferent_ref=interferent_ref,
        seed=303,
        shifted=True,
    )

    x_train = preprocess_spectra(train_set.spectra)
    x_test_id = preprocess_spectra(test_id.spectra)
    x_test_shift = preprocess_spectra(test_shift.spectra)

    peak_idx, peak_shift, sep_score = select_analytical_peak(shifts, target_ref, interferent_ref)
    print(f"Selected analytical peak: {peak_shift:.1f} cm^-1 (separation score={sep_score:.3f})")

    slope, intercept = fit_single_peak_calibration(x_train[:, peak_idx], train_set.c_target)

    ridge = train_ridge_model(x_train, train_set.c_target)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, train_set.c_target, test_size=0.20, random_state=11
    )
    mlp, mlp_scaler, mlp_y_mean, mlp_y_std, best_val = train_torch_mlp(
        x_tr,
        y_tr,
        x_val,
        y_val,
        seed=11,
        epochs=220,
        lr=8e-4,
    )
    print(f"MLP best validation MSE: {best_val:.6f}")

    pred_single_id = predict_single_peak(x_test_id[:, peak_idx], slope, intercept)
    pred_single_shift = predict_single_peak(x_test_shift[:, peak_idx], slope, intercept)

    pred_ridge_id = np.clip(ridge.predict(x_test_id), 0.0, None)
    pred_ridge_shift = np.clip(ridge.predict(x_test_shift), 0.0, None)

    pred_mlp_id = predict_torch_mlp(mlp, mlp_scaler, mlp_y_mean, mlp_y_std, x_test_id)
    pred_mlp_shift = predict_torch_mlp(mlp, mlp_scaler, mlp_y_mean, mlp_y_std, x_test_shift)

    rows = [
        metrics_row("SinglePeak", "ID", test_id.c_target, pred_single_id),
        metrics_row("SinglePeak", "Shifted", test_shift.c_target, pred_single_shift),
        metrics_row("Ridge", "ID", test_id.c_target, pred_ridge_id),
        metrics_row("Ridge", "Shifted", test_shift.c_target, pred_ridge_shift),
        metrics_row("MLP", "ID", test_id.c_target, pred_mlp_id),
        metrics_row("MLP", "Shifted", test_shift.c_target, pred_mlp_shift),
    ]
    result_df = pd.DataFrame(rows)

    print("\n=== Metrics ===")
    print(result_df.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    preview = pd.DataFrame(
        {
            "y_true": test_shift.c_target[:8],
            "single_peak": pred_single_shift[:8],
            "ridge": pred_ridge_shift[:8],
            "mlp": pred_mlp_shift[:8],
        }
    )
    print("\n=== Shifted Sample Predictions (first 8) ===")
    print(preview.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    ridge_id_r2 = float(
        result_df[(result_df["Model"] == "Ridge") & (result_df["Dataset"] == "ID")]["R2"].iloc[0]
    )
    mlp_id_r2 = float(
        result_df[(result_df["Model"] == "MLP") & (result_df["Dataset"] == "ID")]["R2"].iloc[0]
    )
    ridge_shift_r2 = float(
        result_df[(result_df["Model"] == "Ridge") & (result_df["Dataset"] == "Shifted")]["R2"].iloc[0]
    )

    pass_flag = ridge_id_r2 >= 0.90 and mlp_id_r2 >= 0.88 and ridge_shift_r2 >= 0.70
    print(f"\nValidation: {'PASS' if pass_flag else 'FAIL'}")

    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
