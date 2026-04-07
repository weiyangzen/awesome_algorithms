"""Minimal runnable STM MVP.

This script demonstrates a small yet honest algorithmic pipeline for
Scanning Tunneling Microscopy (STM):
1) synthesize a 2D atomic surface and energy-resolved LDOS,
2) generate constant-current topography via Tersoff-Hamann-like model,
3) generate constant-height dI/dV maps,
4) recover lattice spacing from peak detection + DBSCAN clustering,
5) estimate barrier decay constant kappa from I-z spectroscopy,
6) fit a tiny PyTorch regressor from multi-bias spectra to local LDOS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class STMSimConfig:
    n_grid: int = 49
    size_nm: float = 4.2
    lattice_a_nm: float = 0.35
    atom_sigma_nm: float = 0.05
    e_min_ev: float = -0.6
    e_max_ev: float = 0.6
    n_energy: int = 241
    kappa_true: float = 10.8  # nm^-1
    prefactor_c: float = 1.0e-6
    i_setpoint: float = 2.5e-10
    topography_bias_v: float = 0.20
    didv_biases_v: tuple[float, ...] = (0.05, 0.10, 0.20)
    z_const_height_nm: float = 0.55


def make_grid(cfg: STMSimConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, cfg.size_nm, cfg.n_grid)
    y = np.linspace(0.0, cfg.size_nm, cfg.n_grid)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return x, xx, yy


def make_surface_ldos_spatial(cfg: STMSimConfig, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Generate a synthetic atomic lattice image with one vacancy and one impurity."""
    positions = np.arange(cfg.lattice_a_nm / 2.0, cfg.size_nm, cfg.lattice_a_nm)
    lattice = np.zeros_like(xx)

    for px in positions:
        for py in positions:
            lattice += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * cfg.atom_sigma_nm**2))

    vacancy = np.exp(
        -(
            (xx - 0.52 * cfg.size_nm) ** 2
            + (yy - 0.45 * cfg.size_nm) ** 2
        )
        / (2.0 * (0.11**2))
    )
    impurity = np.exp(
        -(
            (xx - 0.73 * cfg.size_nm) ** 2
            + (yy - 0.70 * cfg.size_nm) ** 2
        )
        / (2.0 * (0.09**2))
    )

    spatial = lattice * (1.0 - 0.45 * vacancy) + 0.70 * impurity
    spatial -= spatial.min()
    spatial /= spatial.max() + 1.0e-12
    return spatial


def build_ldos_cube(
    cfg: STMSimConfig,
    spatial: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Build rho(x,y,E) with broad background + localized defect resonance."""
    # Smooth energy trend in the band.
    trend = 1.0 + 0.25 * energies + 0.10 * energies**2
    base = spatial[:, :, None] * trend[None, None, :]

    # A localized resonance near -0.18 eV.
    defect_profile = np.exp(
        -(
            (xx - 0.73 * cfg.size_nm) ** 2
            + (yy - 0.70 * cfg.size_nm) ** 2
        )
        / (2.0 * (0.10**2))
    )
    resonance_e = np.exp(-((energies + 0.18) ** 2) / (2.0 * (0.055**2)))

    rho = 0.12 + base + 0.90 * defect_profile[:, :, None] * resonance_e[None, None, :]
    rho = np.clip(rho, 1.0e-9, None)
    return rho


def integrate_to_bias(rho: np.ndarray, energies: np.ndarray, bias_v: float) -> np.ndarray:
    if bias_v >= 0.0:
        mask = (energies >= 0.0) & (energies <= bias_v)
    else:
        mask = (energies >= bias_v) & (energies <= 0.0)
    e_sel = energies[mask]
    rho_sel = rho[:, :, mask]
    if e_sel.size < 2:
        return np.zeros(rho.shape[:2])
    return trapezoid(rho_sel, e_sel, axis=2)


def simulate_topography(
    cfg: STMSimConfig,
    rho: np.ndarray,
    energies: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-current mode: solve z(x,y) from I_set = C exp(-2kz) * integral LDOS."""
    s_map = integrate_to_bias(rho, energies, cfg.topography_bias_v)
    pref = np.clip(cfg.prefactor_c * s_map, 1.0e-20, None)
    z = (np.log(pref) - np.log(cfg.i_setpoint)) / (2.0 * cfg.kappa_true)
    z += rng.normal(0.0, 0.0025, size=z.shape)
    return z, s_map


def simulate_didv_maps(
    cfg: STMSimConfig,
    rho: np.ndarray,
    energies: np.ndarray,
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """Constant-height differential conductance: dI/dV ∝ exp(-2kz0) * rho(E=eV)."""
    amp = np.exp(-2.0 * cfg.kappa_true * cfg.z_const_height_nm)
    maps: dict[float, np.ndarray] = {}
    for bias in cfg.didv_biases_v:
        idx = int(np.argmin(np.abs(energies - bias)))
        ideal = cfg.prefactor_c * amp * rho[:, :, idx]
        noisy = ideal + rng.normal(0.0, 0.01 * np.std(ideal), size=ideal.shape)
        maps[bias] = np.clip(noisy, 1.0e-16, None)
    return maps


def detect_atoms_and_lattice(
    z_map: np.ndarray,
    x_axis: np.ndarray,
    quantile_threshold: float = 0.93,
) -> tuple[int, float]:
    """Detect atomic maxima and estimate lattice constant from nearest-neighbor spacing."""
    z_smooth = gaussian_filter(z_map, sigma=0.8)
    local_max = z_smooth == maximum_filter(z_smooth, size=3)
    mask = local_max & (z_smooth > np.quantile(z_smooth, quantile_threshold))
    coords = np.argwhere(mask)
    if coords.shape[0] < 2:
        return int(coords.shape[0]), float("nan")

    labels = DBSCAN(eps=1.8, min_samples=1).fit(coords).labels_
    centers_px = []
    for label in np.unique(labels):
        cluster = coords[labels == label]
        centers_px.append(cluster.mean(axis=0))
    centers_px = np.vstack(centers_px)

    centers_nm = np.column_stack(
        (
            np.interp(centers_px[:, 0], np.arange(x_axis.size), x_axis),
            np.interp(centers_px[:, 1], np.arange(x_axis.size), x_axis),
        )
    )

    dmat = cdist(centers_nm, centers_nm)
    np.fill_diagonal(dmat, np.inf)
    nn = np.min(dmat, axis=1)
    lattice_est = float(np.median(nn))
    return int(centers_nm.shape[0]), lattice_est


def estimate_kappa_from_iz(
    cfg: STMSimConfig,
    local_s: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Estimate kappa via ln(I)-z linear fit with sklearn Ridge."""
    z_nm = np.linspace(0.35, 0.80, 80)
    clean_i = cfg.prefactor_c * local_s * np.exp(-2.0 * cfg.kappa_true * z_nm)
    noisy_i = clean_i * (1.0 + rng.normal(0.0, 0.03, size=z_nm.shape))
    noisy_i = np.clip(noisy_i, 1.0e-30, None)

    model = Ridge(alpha=1.0e-8)
    model.fit(z_nm.reshape(-1, 1), np.log(noisy_i))
    slope = float(model.coef_[0])
    kappa_est = -0.5 * slope
    return kappa_est, float(model.intercept_)


def train_torch_ldos_regressor(
    didv_maps: dict[float, np.ndarray],
    target_ldos: np.ndarray,
    seed: int = 7,
) -> tuple[float, float]:
    """Train Ridge and tiny PyTorch MLP to regress local LDOS from multi-bias spectra."""
    bias_keys = sorted(didv_maps.keys())
    features = np.stack([didv_maps[b].ravel() for b in bias_keys], axis=1)
    y = target_ldos.ravel().astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.25, random_state=seed
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_s, y_train)
    ridge_pred = ridge.predict(x_test_s)
    ridge_mae = float(mean_absolute_error(y_test, ridge_pred))

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train_s.shape[1], 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )

    opt = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    loss_fn = torch.nn.MSELoss()

    x_t = torch.tensor(x_train_s, dtype=torch.float32)
    y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    for _ in range(180):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_test = model(torch.tensor(x_test_s, dtype=torch.float32)).squeeze(1).numpy()
    mlp_mae = float(mean_absolute_error(y_test, pred_test))
    return ridge_mae, mlp_mae


def main() -> None:
    seed = 7
    rng = np.random.default_rng(seed)
    cfg = STMSimConfig()

    x_axis, xx, yy = make_grid(cfg)
    energies = np.linspace(cfg.e_min_ev, cfg.e_max_ev, cfg.n_energy)

    spatial = make_surface_ldos_spatial(cfg, xx, yy)
    rho = build_ldos_cube(cfg, spatial, xx, yy, energies)

    topo_z, s_map = simulate_topography(cfg, rho, energies, rng)
    didv_maps = simulate_didv_maps(cfg, rho, energies, rng)

    atom_count, lattice_est = detect_atoms_and_lattice(topo_z, x_axis)

    i_center = cfg.n_grid // 2
    j_center = cfg.n_grid // 2
    kappa_est, intercept = estimate_kappa_from_iz(cfg, float(s_map[i_center, j_center]), rng)

    e0_idx = int(np.argmin(np.abs(energies - 0.0)))
    target_ldos = rho[:, :, e0_idx] / np.max(rho[:, :, e0_idx])
    ridge_mae, mlp_mae = train_torch_ldos_regressor(didv_maps, target_ldos, seed=seed)

    summary = pd.DataFrame(
        [
            {
                "metric": "kappa_true_nm^-1",
                "value": cfg.kappa_true,
            },
            {
                "metric": "kappa_est_nm^-1",
                "value": kappa_est,
            },
            {
                "metric": "kappa_relative_error_%",
                "value": abs(kappa_est - cfg.kappa_true) / cfg.kappa_true * 100.0,
            },
            {
                "metric": "lattice_true_nm",
                "value": cfg.lattice_a_nm,
            },
            {
                "metric": "lattice_est_nm",
                "value": lattice_est,
            },
            {
                "metric": "detected_atomic_sites",
                "value": atom_count,
            },
            {
                "metric": "ridge_mae_ldos",
                "value": ridge_mae,
            },
            {
                "metric": "mlp_mae_ldos",
                "value": mlp_mae,
            },
            {
                "metric": "iz_fit_intercept",
                "value": intercept,
            },
        ]
    )

    didv_stats = pd.DataFrame(
        {
            "bias_V": list(didv_maps.keys()),
            "mean_dIdV": [float(np.mean(didv_maps[b])) for b in didv_maps],
            "std_dIdV": [float(np.std(didv_maps[b])) for b in didv_maps],
        }
    )

    print("=== STM MVP Summary ===")
    print(summary.to_string(index=False))
    print("\n=== dI/dV Map Statistics ===")
    print(didv_stats.to_string(index=False))


if __name__ == "__main__":
    main()
