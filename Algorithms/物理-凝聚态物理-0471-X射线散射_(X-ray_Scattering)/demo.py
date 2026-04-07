"""Minimal runnable MVP for X-ray Scattering in condensed matter.

This demo implements a transparent Debye-scattering workflow:
1) Build finite fcc crystal and amorphous reference structures.
2) Compute powder-averaged static structure factor S(q) via Debye equation.
3) Detect Bragg-like peaks and compare with fcc theoretical q positions.
4) Use sklearn for peak-position regression diagnostics.
5) Use PyTorch to inversely fit lattice constant and Debye-Waller factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class XRayConfig:
    """Configuration for a deterministic, small, auditable MVP."""

    lattice_constant_angstrom: float = 3.60
    n_cells: int = 3
    q_min_inv_angstrom: float = 0.15
    q_max_inv_angstrom: float = 12.0
    n_q: int = 380
    debye_waller_true: float = 0.015
    gaussian_noise_std: float = 0.03
    smoothing_sigma: float = 1.2
    peak_prominence: float = 0.14
    bragg_q_min_inv_angstrom: float = 2.2
    peak_match_tolerance_inv_angstrom: float = 0.30
    torch_epochs: int = 260
    torch_lr: float = 0.05
    random_seed: int = 7


def fcc_unit_basis() -> np.ndarray:
    """Return fractional basis coordinates for monoatomic fcc conventional cell."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )


def generate_fcc_positions(n_cells: int, lattice_constant: float) -> np.ndarray:
    """Generate finite fcc supercell cartesian coordinates in Angstrom."""
    basis = fcc_unit_basis()
    coords: List[np.ndarray] = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                cell_origin = np.array([ix, iy, iz], dtype=np.float64)
                frac = basis + cell_origin
                coords.append(frac * lattice_constant)
    return np.vstack(coords)


def generate_amorphous_positions(
    n_atoms: int,
    box_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random uniform point cloud as amorphous reference."""
    return rng.uniform(0.0, box_length, size=(n_atoms, 3)).astype(np.float64)


def pair_distances_upper(coords: np.ndarray) -> np.ndarray:
    """Return all pair distances r_ij (i<j) for Debye equation."""
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(coords.shape[0], k=1)
    return distances[iu]


def debye_structure_factor(
    q_values: np.ndarray,
    pair_distances: np.ndarray,
    n_atoms: int,
    debye_waller_b: float = 0.0,
) -> np.ndarray:
    """Compute powder static structure factor S(q) by Debye scattering equation.

    S(q) = 1 + (2/N) * sum_{i<j} sin(q r_ij)/(q r_ij),
    with an optional Debye-Waller damping on coherent term:
    S_dw(q) = 1 + (S(q)-1) * exp(-B q^2).
    """
    qr = np.outer(q_values, pair_distances)
    sinc_term = np.sinc(qr / np.pi)  # np.sinc(x)=sin(pi x)/(pi x)
    s_q = 1.0 + (2.0 / float(n_atoms)) * np.sum(sinc_term, axis=1)
    if debye_waller_b > 0.0:
        damping = np.exp(-debye_waller_b * q_values * q_values)
        s_q = 1.0 + (s_q - 1.0) * damping
    return s_q


def first_fcc_reflections(max_index: int, lattice_constant: float, n_take: int) -> np.ndarray:
    """Return first n_take theoretical powder peak positions q for monoatomic fcc."""
    s2_values = set()
    for h in range(0, max_index + 1):
        for k in range(0, max_index + 1):
            for l in range(0, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                same_parity = (h % 2 == k % 2 == l % 2)
                if not same_parity:
                    continue
                s2_values.add(h * h + k * k + l * l)
    s2_sorted = np.array(sorted(s2_values), dtype=np.float64)
    q_values = (2.0 * np.pi / lattice_constant) * np.sqrt(s2_sorted)
    return q_values[:n_take]


def match_peaks_to_theory(
    detected_q: np.ndarray,
    theory_q: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each detected peak to nearest theory peak within tolerance."""
    matched_theory: List[float] = []
    matched_detected: List[float] = []
    matched_error: List[float] = []
    for q_det in detected_q:
        idx = int(np.argmin(np.abs(theory_q - q_det)))
        q_ref = float(theory_q[idx])
        err = abs(float(q_det) - q_ref)
        if err <= tolerance:
            matched_theory.append(q_ref)
            matched_detected.append(float(q_det))
            matched_error.append(err)
    return (
        np.asarray(matched_theory, dtype=np.float64),
        np.asarray(matched_detected, dtype=np.float64),
        np.asarray(matched_error, dtype=np.float64),
    )


def torch_fit_lattice_and_dw(
    q_values: np.ndarray,
    pair_distances_unit_a: np.ndarray,
    n_atoms: int,
    target_sq: np.ndarray,
    initial_a: float,
    epochs: int,
    lr: float,
) -> Tuple[float, float, float]:
    """Fit lattice constant and Debye-Waller factor with a differentiable Debye model."""
    q_t = torch.tensor(q_values, dtype=torch.float32)
    d_unit_t = torch.tensor(pair_distances_unit_a, dtype=torch.float32)
    target_t = torch.tensor(target_sq, dtype=torch.float32)

    a_raw = torch.nn.Parameter(torch.tensor(np.log(np.exp(initial_a - 1.0) - 1.0), dtype=torch.float32))
    b_raw = torch.nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
    optimizer = torch.optim.Adam([a_raw, b_raw], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        a = torch.nn.functional.softplus(a_raw) + 1.0
        b = torch.nn.functional.softplus(b_raw)

        qr = q_t[:, None] * (a * d_unit_t[None, :])
        small = qr.abs() < 1e-6
        sinc_term = torch.where(small, torch.ones_like(qr), torch.sin(qr) / qr)
        s_q = 1.0 + (2.0 / float(n_atoms)) * torch.sum(sinc_term, dim=1)
        s_q = 1.0 + (s_q - 1.0) * torch.exp(-b * q_t * q_t)

        loss = torch.mean((s_q - target_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        a_fit = float(torch.nn.functional.softplus(a_raw).item() + 1.0)
        b_fit = float(torch.nn.functional.softplus(b_raw).item())
        qr = q_t[:, None] * (a_fit * d_unit_t[None, :])
        sinc_term = torch.where(qr.abs() < 1e-6, torch.ones_like(qr), torch.sin(qr) / qr)
        s_q_fit = 1.0 + (2.0 / float(n_atoms)) * torch.sum(sinc_term, dim=1)
        s_q_fit = 1.0 + (s_q_fit - 1.0) * torch.exp(-b_fit * q_t * q_t)
        mse = float(torch.mean((s_q_fit - target_t) ** 2).item())
    return a_fit, b_fit, mse


def main() -> None:
    cfg = XRayConfig()
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    rng = np.random.default_rng(cfg.random_seed)

    q_values = np.linspace(cfg.q_min_inv_angstrom, cfg.q_max_inv_angstrom, cfg.n_q, dtype=np.float64)
    crystal = generate_fcc_positions(cfg.n_cells, cfg.lattice_constant_angstrom)
    n_atoms = crystal.shape[0]
    box_length = cfg.n_cells * cfg.lattice_constant_angstrom

    crystal_pair_dist = pair_distances_upper(crystal)
    sq_crystal_clean = debye_structure_factor(
        q_values=q_values,
        pair_distances=crystal_pair_dist,
        n_atoms=n_atoms,
        debye_waller_b=cfg.debye_waller_true,
    )

    sq_crystal_noisy = sq_crystal_clean + rng.normal(0.0, cfg.gaussian_noise_std, size=cfg.n_q)
    sq_crystal_noisy = np.clip(sq_crystal_noisy, 0.0, None)
    sq_crystal_smooth = gaussian_filter1d(sq_crystal_noisy, sigma=cfg.smoothing_sigma)

    amorphous = generate_amorphous_positions(n_atoms=n_atoms, box_length=box_length, rng=rng)
    amorphous_pair_dist = pair_distances_upper(amorphous)
    sq_amorphous = debye_structure_factor(q_values, amorphous_pair_dist, n_atoms, debye_waller_b=0.0)
    sq_amorphous_smooth = gaussian_filter1d(sq_amorphous, sigma=cfg.smoothing_sigma)

    peak_indices_all, peak_props_all = find_peaks(sq_crystal_smooth, prominence=cfg.peak_prominence)
    peak_mask = q_values[peak_indices_all] >= cfg.bragg_q_min_inv_angstrom
    peak_indices = peak_indices_all[peak_mask]
    peak_prominence = peak_props_all["prominences"][peak_mask]
    peak_q = q_values[peak_indices]
    peak_intensity = sq_crystal_smooth[peak_indices]

    peak_indices_amorphous_all, peak_props_amorphous_all = find_peaks(
        sq_amorphous_smooth,
        prominence=cfg.peak_prominence,
    )
    peak_mask_amorphous = q_values[peak_indices_amorphous_all] >= cfg.bragg_q_min_inv_angstrom
    peak_indices_amorphous = peak_indices_amorphous_all[peak_mask_amorphous]
    peak_prominence_amorphous = peak_props_amorphous_all["prominences"][peak_mask_amorphous]

    theoretical_q = first_fcc_reflections(max_index=10, lattice_constant=cfg.lattice_constant_angstrom, n_take=24)
    matched_theory_q, matched_detected_q, matched_error = match_peaks_to_theory(
        detected_q=peak_q,
        theory_q=theoretical_q,
        tolerance=cfg.peak_match_tolerance_inv_angstrom,
    )

    if len(matched_detected_q) < 3:
        raise RuntimeError("Too few matched peaks for regression diagnostics.")

    x = matched_theory_q.reshape(-1, 1)
    y = matched_detected_q
    reg = LinearRegression().fit(x, y)
    r2 = float(reg.score(x, y))
    rmse = float(np.sqrt(mean_squared_error(y, reg.predict(x))))

    crystal_unit = generate_fcc_positions(cfg.n_cells, lattice_constant=1.0)
    unit_pair_dist = pair_distances_upper(crystal_unit)
    a_fit, b_fit, torch_mse = torch_fit_lattice_and_dw(
        q_values=q_values,
        pair_distances_unit_a=unit_pair_dist,
        n_atoms=n_atoms,
        target_sq=sq_crystal_smooth,
        initial_a=cfg.lattice_constant_angstrom * 0.90,
        epochs=cfg.torch_epochs,
        lr=cfg.torch_lr,
    )

    peak_table = pd.DataFrame(
        {
            "q_peak_invA": peak_q,
            "S_peak": peak_intensity,
            "prominence": peak_prominence,
        }
    ).head(8)

    compare_count = min(8, len(matched_detected_q))
    match_table = pd.DataFrame(
        {
            "q_theory_invA": matched_theory_q[:compare_count],
            "q_detected_invA": matched_detected_q[:compare_count],
            "abs_error": matched_error[:compare_count],
        }
    )

    q111 = (2.0 * np.pi / cfg.lattice_constant_angstrom) * np.sqrt(3.0)
    first_peak_error = float(np.min(np.abs(peak_q - q111)))
    crystal_prom_sum = float(np.sum(peak_prominence))
    amorphous_prom_sum = float(np.sum(peak_prominence_amorphous))

    assert first_peak_error < 0.15, f"First Bragg-peak position error too large: {first_peak_error:.4f}"
    assert len(peak_q) >= 4, "Detected too few crystal peaks in Bragg region."
    assert crystal_prom_sum > 6.0 * max(amorphous_prom_sum, 1e-8), "Crystal peak prominence is not sufficiently stronger than amorphous."
    assert r2 > 0.995, f"Peak regression R^2 too low: {r2:.5f}"
    assert abs(a_fit - cfg.lattice_constant_angstrom) < 0.10, "Torch inversion failed to recover lattice constant."
    assert torch_mse < 0.10, f"Torch inversion MSE too large: {torch_mse:.5f}"

    print("=== X-ray Scattering MVP (Debye equation, powder profile) ===")
    print(f"atoms_in_supercell      : {n_atoms}")
    print(f"lattice_constant_true_A : {cfg.lattice_constant_angstrom:.4f}")
    print(f"debye_waller_true       : {cfg.debye_waller_true:.5f}")
    print(f"q_range_invA            : [{cfg.q_min_inv_angstrom:.3f}, {cfg.q_max_inv_angstrom:.3f}]")
    print(f"num_detected_peaks      : {len(peak_q)}")
    print(f"num_detected_peaks_amor : {len(peak_indices_amorphous)}")

    print("\nDetected peak summary (top 8 by q order):")
    peak_fmt = peak_table.copy()
    for col in peak_fmt.columns:
        peak_fmt[col] = peak_fmt[col].map(lambda v: f"{float(v):.5f}")
    print(peak_fmt.to_string(index=False))

    print("\nTheory vs detected (nearest matched peaks):")
    match_fmt = match_table.copy()
    for col in match_fmt.columns:
        match_fmt[col] = match_fmt[col].map(lambda v: f"{float(v):.5f}")
    print(match_fmt.to_string(index=False))

    print("\nDiagnostics:")
    print(f"first_peak_error_invA   : {first_peak_error:.6f}")
    print(f"peak_regression_R2      : {r2:.6f}")
    print(f"peak_regression_RMSE    : {rmse:.6f}")
    print(f"torch_lattice_fit_A     : {a_fit:.6f}")
    print(f"torch_dw_fit            : {b_fit:.6f}")
    print(f"torch_fit_mse           : {torch_mse:.6f}")
    print(f"crystal_prominence_sum  : {crystal_prom_sum:.6f}")
    print(f"amorphous_prominence_sum: {amorphous_prom_sum:.6f}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
