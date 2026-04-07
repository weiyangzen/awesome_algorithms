"""Minimal runnable MVP for infrared spectroscopy (FTIR) concentration estimation.

Pipeline:
1) Synthesize FTIR absorbance spectra with analyte/interferent peaks plus baseline/scatter/noise.
2) Preprocess with Savitzky-Golay smoothing + AsLS baseline correction + vector normalization.
3) Estimate analyte concentration by a single analytical band (Beer-Lambert style).
4) Compare with full-spectrum Ridge regression and a small PyTorch MLP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class IRDataset:
    """Container for simulated IR spectra and concentration labels."""

    spectra: np.ndarray
    c_analyte: np.ndarray
    c_interferent: np.ndarray


class SmallMLP(nn.Module):
    """Small nonlinear regressor for full-spectrum modeling."""

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


def gaussian_band(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Return a Gaussian-like IR absorption band profile."""
    z = (x - center) / width
    return np.exp(-0.5 * z * z)


def ir_absorptivity_profiles(wavenumbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic analyte/interferent absorptivity curves.

    Wavenumber unit: cm^-1.
    """
    # Example analyte: strong C=O around 1720 cm^-1 + C-H stretches around 2920 cm^-1.
    eps_a = (
        1.40 * gaussian_band(wavenumbers, 1720.0, 30.0)
        + 0.75 * gaussian_band(wavenumbers, 2925.0, 38.0)
        + 0.35 * gaussian_band(wavenumbers, 1455.0, 45.0)
    )

    # Example interferent: broad O-H/N-H + amide-like region overlap.
    eps_b = (
        1.00 * gaussian_band(wavenumbers, 1650.0, 42.0)
        + 0.85 * gaussian_band(wavenumbers, 1540.0, 55.0)
        + 0.55 * gaussian_band(wavenumbers, 3400.0, 120.0)
        + 0.20 * gaussian_band(wavenumbers, 1100.0, 75.0)
    )

    return eps_a, eps_b


def simulate_ftir_dataset(
    n_samples: int,
    wavenumbers: np.ndarray,
    eps_a: np.ndarray,
    eps_b: np.ndarray,
    seed: int,
    shift: bool,
) -> IRDataset:
    """Generate FTIR-like spectra with realistic distortions.

    If shift=True, increase interferent strength/noise/scatter to emulate distribution shift.
    """
    rng = np.random.default_rng(seed)
    n_wn = wavenumbers.size

    c_analyte = rng.uniform(0.05, 1.30, size=n_samples)
    if shift:
        c_interferent = rng.uniform(0.20, 1.10, size=n_samples)
        noise_sigma = 0.010
        scatter_sigma = 0.040
    else:
        c_interferent = rng.uniform(0.00, 0.80, size=n_samples)
        noise_sigma = 0.006
        scatter_sigma = 0.022

    absorbance_core = np.outer(c_analyte, eps_a) + np.outer(c_interferent, eps_b)

    # Multiplicative scatter / path-length-like fluctuation.
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, 1))
    wn_phase = (wavenumbers[None, :] - wavenumbers.min()) / 250.0
    multiplicative = 1.0 + rng.normal(0.0, scatter_sigma, size=(n_samples, 1)) * np.sin(
        wn_phase + phase
    )

    # Smooth baseline drift (polynomial).
    x = (wavenumbers - wavenumbers.mean()) / (0.5 * (wavenumbers.max() - wavenumbers.min()))
    b0 = rng.normal(0.0, 0.009 if shift else 0.005, size=(n_samples, 1))
    b1 = rng.normal(0.0, 0.020 if shift else 0.012, size=(n_samples, 1))
    b2 = rng.normal(0.0, 0.018 if shift else 0.010, size=(n_samples, 1))
    baseline = b0 + b1 * x[None, :] + b2 * (x[None, :] ** 2)

    # Water-vapor-like artifact region around 3600 cm^-1.
    water_scale = rng.uniform(0.0, 0.05 if shift else 0.03, size=(n_samples, 1))
    water_band = water_scale * gaussian_band(wavenumbers[None, :], 3600.0, 95.0)

    noise = rng.normal(0.0, noise_sigma, size=(n_samples, n_wn))

    spectra = absorbance_core * multiplicative + baseline + water_band + noise
    return IRDataset(spectra=spectra, c_analyte=c_analyte, c_interferent=c_interferent)


def make_dtd_matrix(n_points: int) -> sparse.csc_matrix:
    """Construct D^T D for second-order difference operator used by AsLS."""
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_points - 2, n_points), format="csc")
    return (d.T @ d).tocsc()


def asls_baseline(
    y: np.ndarray,
    dtd: sparse.csc_matrix,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> np.ndarray:
    """Asymmetric least squares baseline correction (Eilers-style)."""
    n = y.size
    w = np.ones(n)

    for _ in range(n_iter):
        w_mat = sparse.spdiags(w, 0, n, n)
        z = spsolve(w_mat + lam * dtd, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)

    return np.asarray(z)


def preprocess_spectra(spectra: np.ndarray) -> np.ndarray:
    """Smooth, baseline-correct, and normalize FTIR spectra."""
    smoothed = savgol_filter(spectra, window_length=15, polyorder=3, axis=1, mode="interp")

    n_samples, n_wn = smoothed.shape
    dtd = make_dtd_matrix(n_wn)
    corrected = np.empty_like(smoothed)

    for i in range(n_samples):
        baseline = asls_baseline(smoothed[i], dtd=dtd, lam=1.2e5, p=0.02, n_iter=10)
        row = smoothed[i] - baseline
        row_norm = np.linalg.norm(row) + 1e-12
        corrected[i] = row / row_norm

    return corrected


def select_analytical_band(
    wavenumbers: np.ndarray, eps_a: np.ndarray, eps_b: np.ndarray
) -> tuple[int, float]:
    """Select a high-separability analyte band index (Beer-Lambert style)."""
    peaks, _ = find_peaks(eps_a, prominence=0.08 * float(np.max(eps_a)), distance=10)

    if peaks.size == 0:
        idx = int(np.argmax(eps_a))
        return idx, float(wavenumbers[idx])

    denominator = eps_b[peaks] + 0.08 * float(np.max(eps_b)) + 1e-12
    score = eps_a[peaks] / denominator
    best_idx = int(peaks[int(np.argmax(score))])
    return best_idx, float(wavenumbers[best_idx])


def beer_lambert_estimate(
    corrected_spectra: np.ndarray, band_idx: int, eps_a: np.ndarray, path_length_cm: float = 1.0
) -> np.ndarray:
    """Single-band concentration estimate by Beer-Lambert relation."""
    denom = max(float(eps_a[band_idx]) * path_length_cm, 1e-12)
    pred = corrected_spectra[:, band_idx] / denom
    return np.clip(pred, 0.0, None)


def train_ridge_model(x_train: np.ndarray, y_train: np.ndarray) -> object:
    """Train scaled Ridge regressor for full-spectrum regression."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    return model


def train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
    epochs: int = 140,
    lr: float = 1e-3,
) -> tuple[SmallMLP, StandardScaler, float]:
    """Train compact MLP regressor and keep best validation checkpoint."""
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(x_train)
    x_train_std = scaler.transform(x_train).astype(np.float32)
    x_val_std = scaler.transform(x_val).astype(np.float32)
    y_train_f = y_train.astype(np.float32).reshape(-1, 1)
    y_val_f = y_val.astype(np.float32).reshape(-1, 1)

    train_ds = TensorDataset(torch.from_numpy(x_train_std), torch.from_numpy(y_train_f))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = SmallMLP(in_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_val_t = torch.from_numpy(x_val_std)
    y_val_t = torch.from_numpy(y_val_f)

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
            val_pred = model(x_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, best_val


def predict_torch_mlp(model: SmallMLP, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    """Run MLP prediction on new spectra."""
    model.eval()
    x_std = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(x_std)).squeeze(1).cpu().numpy()
    return np.clip(pred, 0.0, None)


def metrics_row(model_name: str, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for display."""
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
    np.random.seed(2026)
    torch.manual_seed(2026)

    # FTIR convention often displayed high->low wavenumber.
    wavenumbers = np.linspace(4000.0, 650.0, num=701)
    eps_a, eps_b = ir_absorptivity_profiles(wavenumbers)

    train_ds = simulate_ftir_dataset(
        n_samples=620,
        wavenumbers=wavenumbers,
        eps_a=eps_a,
        eps_b=eps_b,
        seed=42,
        shift=False,
    )
    id_test_ds = simulate_ftir_dataset(
        n_samples=220,
        wavenumbers=wavenumbers,
        eps_a=eps_a,
        eps_b=eps_b,
        seed=123,
        shift=False,
    )
    shifted_test_ds = simulate_ftir_dataset(
        n_samples=220,
        wavenumbers=wavenumbers,
        eps_a=eps_a,
        eps_b=eps_b,
        seed=321,
        shift=True,
    )

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        train_ds.spectra,
        train_ds.c_analyte,
        test_size=0.2,
        random_state=2026,
    )

    x_train = preprocess_spectra(x_train_raw)
    x_val = preprocess_spectra(x_val_raw)
    x_id = preprocess_spectra(id_test_ds.spectra)
    x_shift = preprocess_spectra(shifted_test_ds.spectra)

    band_idx, band_wn = select_analytical_band(wavenumbers, eps_a, eps_b)

    y_beer_id = beer_lambert_estimate(x_id, band_idx=band_idx, eps_a=eps_a)
    y_beer_shift = beer_lambert_estimate(x_shift, band_idx=band_idx, eps_a=eps_a)

    ridge = train_ridge_model(x_train, y_train)
    y_ridge_id = np.clip(ridge.predict(x_id), 0.0, None)
    y_ridge_shift = np.clip(ridge.predict(x_shift), 0.0, None)

    mlp, mlp_scaler, best_val_loss = train_torch_mlp(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        seed=7,
        epochs=140,
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

    print(f"Selected analytical band: {band_wn:.1f} cm^-1 (index={band_idx})")
    print(f"Best MLP validation MSE: {best_val_loss:.6f}")
    print("\nFTIR concentration regression benchmark")
    print(result_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    preview = pd.DataFrame(
        {
            "true": shifted_test_ds.c_analyte[:8],
            "beer": y_beer_shift[:8],
            "ridge": y_ridge_shift[:8],
            "mlp": y_mlp_shift[:8],
        }
    )
    print("\nShifted-set prediction preview (first 8 rows)")
    print(preview.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ridge_id_r2 = float(
        result_df[(result_df["Model"] == "Ridge") & (result_df["Dataset"] == "ID")]["R2"].iloc[0]
    )
    beer_id_r2 = float(
        result_df[(result_df["Model"] == "Beer-Lambert") & (result_df["Dataset"] == "ID")]["R2"].iloc[0]
    )

    assert np.isfinite(result_df[["MAE", "RMSE", "R2"]].to_numpy()).all(), "Non-finite metric detected"
    assert 1200.0 <= band_wn <= 3100.0, "Selected IR band is outside expected analyte region"
    assert ridge_id_r2 > beer_id_r2, "Ridge should beat single-band baseline on ID set"
    assert ridge_id_r2 > 0.25, "Ridge fit quality is unexpectedly low for this synthetic setting"


if __name__ == "__main__":
    main()
