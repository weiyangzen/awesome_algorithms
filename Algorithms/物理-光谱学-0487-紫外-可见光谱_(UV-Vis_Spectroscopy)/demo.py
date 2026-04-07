"""Minimal runnable MVP for UV-Vis spectroscopy concentration estimation.

Pipeline:
1) Synthesize UV-Vis spectra with analyte + interferent + baseline/noise.
2) Preprocess spectra (Savitzky-Golay smoothing + polynomial baseline correction).
3) Estimate analyte concentration via Beer-Lambert at an automatically selected wavelength.
4) Compare against multivariate Ridge regression and a small PyTorch MLP.
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
class UVVisDataset:
    """Container for simulated UV-Vis spectra and concentrations."""

    spectra: np.ndarray
    c_analyte: np.ndarray
    c_interferent: np.ndarray


class SmallMLP(nn.Module):
    """Tiny regressor used as a nonlinear UV-Vis baseline."""

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


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian peak shape."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z)


def extinction_profiles(wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct synthetic molar absorptivity profiles for analyte/interferent."""
    eps_a = 1.20 * gaussian(wavelengths, 278.0, 16.0) + 0.85 * gaussian(
        wavelengths, 515.0, 33.0
    )
    eps_b = (
        0.70 * gaussian(wavelengths, 305.0, 22.0)
        + 1.00 * gaussian(wavelengths, 545.0, 25.0)
        + 0.25 * gaussian(wavelengths, 660.0, 42.0)
    )
    return eps_a, eps_b


def simulate_uvvis_dataset(
    n_samples: int,
    wavelengths: np.ndarray,
    eps_a: np.ndarray,
    eps_b: np.ndarray,
    seed: int,
    shift: bool,
) -> UVVisDataset:
    """Generate spectra with baseline drift, scattering, and noise.

    If shift=True, create a slightly harder distribution (higher interferent and noise).
    """
    rng = np.random.default_rng(seed)
    n_wl = wavelengths.size

    c_analyte = rng.uniform(0.05, 1.20, size=n_samples)
    if shift:
        c_interferent = rng.uniform(0.15, 1.00, size=n_samples)
        noise_sigma = 0.010
        warp_scale = 0.040
    else:
        c_interferent = rng.uniform(0.00, 0.70, size=n_samples)
        noise_sigma = 0.005
        warp_scale = 0.020

    absorbance_core = np.outer(c_analyte, eps_a) + np.outer(c_interferent, eps_b)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, 1))
    wavelength_phase = (wavelengths[None, :] - wavelengths.min()) / 80.0
    warp = 1.0 + rng.normal(0.0, warp_scale, size=(n_samples, 1)) * np.sin(
        wavelength_phase + phase
    )

    x = (wavelengths - wavelengths.mean()) / (0.5 * (wavelengths.max() - wavelengths.min()))
    b0 = rng.normal(0.0, 0.010 if shift else 0.006, size=(n_samples, 1))
    b1 = rng.normal(0.0, 0.025 if shift else 0.015, size=(n_samples, 1))
    b2 = rng.normal(0.0, 0.018 if shift else 0.010, size=(n_samples, 1))
    baseline = b0 + b1 * x[None, :] + b2 * (x[None, :] ** 2)

    scattering_strength = rng.uniform(0.0, 0.020 if shift else 0.010, size=(n_samples, 1))
    scattering = scattering_strength * ((400.0 / wavelengths[None, :]) ** 4)

    noise = rng.normal(0.0, noise_sigma, size=(n_samples, n_wl))

    spectra = absorbance_core * warp + baseline + scattering + noise
    return UVVisDataset(spectra=spectra, c_analyte=c_analyte, c_interferent=c_interferent)


def preprocess_spectra(spectra: np.ndarray) -> np.ndarray:
    """Apply smoothing and polynomial baseline correction per spectrum."""
    smoothed = savgol_filter(spectra, window_length=11, polyorder=3, axis=1, mode="interp")

    n_samples, n_wl = smoothed.shape
    x = np.linspace(-1.0, 1.0, num=n_wl)
    corrected = np.empty_like(smoothed)

    for i in range(n_samples):
        row = smoothed[i]
        q = np.quantile(row, 0.35)
        mask = row <= q

        if np.count_nonzero(mask) < 12:
            idx = np.argsort(row)[:12]
            mask = np.zeros_like(row, dtype=bool)
            mask[idx] = True

        coeff = np.polyfit(x[mask], row[mask], deg=2)
        baseline = np.polyval(coeff, x)
        corrected[i] = row - baseline

    return corrected


def select_analytical_wavelength(
    wavelengths: np.ndarray, eps_a: np.ndarray, eps_b: np.ndarray
) -> tuple[int, float]:
    """Find an analyte peak with high analyte/interferent separation."""
    peaks, _ = find_peaks(eps_a, prominence=0.08 * float(np.max(eps_a)), distance=15)

    if peaks.size == 0:
        idx = int(np.argmax(eps_a))
        return idx, float(wavelengths[idx])

    denominator = eps_b[peaks] + 0.05 * float(np.max(eps_b)) + 1e-12
    score = eps_a[peaks] / denominator
    best = int(peaks[int(np.argmax(score))])
    return best, float(wavelengths[best])


def beer_lambert_estimate(
    corrected_spectra: np.ndarray, peak_idx: int, eps_a: np.ndarray, path_length_cm: float = 1.0
) -> np.ndarray:
    """Single-wavelength Beer-Lambert concentration estimator."""
    denom = max(float(eps_a[peak_idx]) * path_length_cm, 1e-12)
    c_hat = corrected_spectra[:, peak_idx] / denom
    return np.clip(c_hat, 0.0, None)


def train_ridge_model(x_train: np.ndarray, y_train: np.ndarray) -> object:
    """Train a scaled Ridge regressor on full spectra."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.2))
    model.fit(x_train, y_train)
    return model


def train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
    epochs: int = 120,
    lr: float = 1e-3,
) -> tuple[SmallMLP, StandardScaler, float]:
    """Train a small MLP on standardized spectra."""
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(x_train)
    x_train_std = scaler.transform(x_train).astype(np.float32)
    x_val_std = scaler.transform(x_val).astype(np.float32)
    y_train_f = y_train.astype(np.float32).reshape(-1, 1)
    y_val_f = y_val.astype(np.float32).reshape(-1, 1)

    dataset = TensorDataset(torch.from_numpy(x_train_std), torch.from_numpy(y_train_f))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SmallMLP(in_dim=x_train_std.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_val_tensor = torch.from_numpy(x_val_std)
    y_val_tensor = torch.from_numpy(y_val_f)

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

    return model, scaler, best_val


def predict_torch_mlp(model: SmallMLP, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    """Predict concentration with trained MLP."""
    model.eval()
    x_std = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(x_std)).squeeze(1).cpu().numpy()
    return np.clip(pred, 0.0, None)


def metrics_row(model_name: str, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics in a printable row."""
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
    np.random.seed(7)
    torch.manual_seed(7)

    wavelengths = np.arange(220.0, 801.0, 1.0)
    eps_a, eps_b = extinction_profiles(wavelengths)

    train_ds = simulate_uvvis_dataset(
        n_samples=1000,
        wavelengths=wavelengths,
        eps_a=eps_a,
        eps_b=eps_b,
        seed=42,
        shift=False,
    )
    id_test_ds = simulate_uvvis_dataset(
        n_samples=260,
        wavelengths=wavelengths,
        eps_a=eps_a,
        eps_b=eps_b,
        seed=123,
        shift=False,
    )
    shifted_test_ds = simulate_uvvis_dataset(
        n_samples=260,
        wavelengths=wavelengths,
        eps_a=eps_a,
        eps_b=eps_b,
        seed=321,
        shift=True,
    )

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        train_ds.spectra, train_ds.c_analyte, test_size=0.2, random_state=2026
    )

    x_train = preprocess_spectra(x_train_raw)
    x_val = preprocess_spectra(x_val_raw)
    x_id = preprocess_spectra(id_test_ds.spectra)
    x_shift = preprocess_spectra(shifted_test_ds.spectra)

    peak_idx, peak_wl = select_analytical_wavelength(wavelengths, eps_a, eps_b)

    y_beer_id = beer_lambert_estimate(x_id, peak_idx, eps_a)
    y_beer_shift = beer_lambert_estimate(x_shift, peak_idx, eps_a)

    ridge = train_ridge_model(x_train, y_train)
    y_ridge_id = np.clip(ridge.predict(x_id), 0.0, None)
    y_ridge_shift = np.clip(ridge.predict(x_shift), 0.0, None)

    mlp, mlp_scaler, best_val_loss = train_torch_mlp(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        seed=19,
        epochs=120,
        lr=1e-3,
    )
    y_mlp_id = predict_torch_mlp(mlp, mlp_scaler, x_id)
    y_mlp_shift = predict_torch_mlp(mlp, mlp_scaler, x_shift)

    rows = [
        metrics_row("Beer-Lambert", "ID", id_test_ds.c_analyte, y_beer_id),
        metrics_row("Beer-Lambert", "Shifted", shifted_test_ds.c_analyte, y_beer_shift),
        metrics_row("Ridge", "ID", id_test_ds.c_analyte, y_ridge_id),
        metrics_row("Ridge", "Shifted", shifted_test_ds.c_analyte, y_ridge_shift),
        metrics_row("Torch-MLP", "ID", id_test_ds.c_analyte, y_mlp_id),
        metrics_row("Torch-MLP", "Shifted", shifted_test_ds.c_analyte, y_mlp_shift),
    ]
    result_df = pd.DataFrame(rows)

    print(f"Selected analytical wavelength: {peak_wl:.1f} nm (index={peak_idx})")
    print(f"Best MLP validation MSE: {best_val_loss:.6f}")
    print("\nUV-Vis concentration regression benchmark")
    print(result_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    preview = pd.DataFrame(
        {
            "true": shifted_test_ds.c_analyte[:8],
            "beer": y_beer_shift[:8],
            "ridge": y_ridge_shift[:8],
            "mlp": y_mlp_shift[:8],
        }
    )
    print("\nShifted set prediction preview (first 8 rows)")
    print(preview.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ridge_id_r2 = float(
        result_df[(result_df["Model"] == "Ridge") & (result_df["Dataset"] == "ID")]["R2"].iloc[0]
    )
    beer_id_r2 = float(
        result_df[(result_df["Model"] == "Beer-Lambert") & (result_df["Dataset"] == "ID")]["R2"].iloc[0]
    )

    assert np.isfinite(result_df[["MAE", "RMSE", "R2"]].to_numpy()).all(), "Non-finite metric detected"
    assert 250.0 <= peak_wl <= 560.0, "Selected wavelength is outside expected analyte peak region"
    assert ridge_id_r2 > beer_id_r2, "Ridge should outperform single-wavelength baseline on ID set"
    assert ridge_id_r2 > 0.85, "Ridge fit quality is unexpectedly low"


if __name__ == "__main__":
    main()
