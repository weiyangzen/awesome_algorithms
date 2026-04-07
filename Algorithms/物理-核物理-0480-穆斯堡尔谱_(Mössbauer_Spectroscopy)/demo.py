"""Minimal runnable MVP for Mössbauer Spectroscopy.

This script builds a transparent end-to-end pipeline:
1) synthesize a Fe-57-like sextet spectrum in velocity space,
2) fit hyperfine parameters via SciPy nonlinear least squares,
3) score fit quality via scikit-learn metrics,
4) optionally run a short PyTorch gradient refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - runtime fallback only
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

# Powder-like sextet line intensities for Fe-57 magnetic splitting.
OFFSETS = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], dtype=float)
REL_INTENSITY = np.array([3.0, 2.0, 1.0, 1.0, 2.0, 3.0], dtype=float)
REL_INTENSITY = REL_INTENSITY / REL_INTENSITY.sum()


@dataclass(frozen=True)
class SextetParams:
    """Parameterization of a symmetric Mössbauer sextet profile."""

    baseline: float
    depth: float
    isomer_shift: float
    hyperfine_step: float
    linewidth: float


def lorentzian(v: np.ndarray, center: float, gamma: float) -> np.ndarray:
    """Peak shape with unit amplitude at resonance center."""

    return (gamma**2) / ((v - center) ** 2 + gamma**2)


def mossbauer_sextet_numpy(
    velocity: np.ndarray,
    baseline: float,
    depth: float,
    isomer_shift: float,
    hyperfine_step: float,
    linewidth: float,
) -> np.ndarray:
    """Six-line transmission model in Doppler velocity space (mm/s)."""

    centers = isomer_shift + hyperfine_step * OFFSETS
    profile = np.zeros_like(velocity, dtype=float)
    for weight, center in zip(REL_INTENSITY, centers):
        profile += weight * lorentzian(velocity, center, linewidth)
    return baseline - depth * profile


def generate_synthetic_data(
    true_params: SextetParams,
    n_points: int = 401,
    v_min: float = -6.0,
    v_max: float = 6.0,
    noise_std: float = 0.004,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic spectrum for deterministic MVP validation."""

    rng = np.random.default_rng(seed)
    velocity = np.linspace(v_min, v_max, n_points, dtype=float)
    clean = mossbauer_sextet_numpy(
        velocity,
        true_params.baseline,
        true_params.depth,
        true_params.isomer_shift,
        true_params.hyperfine_step,
        true_params.linewidth,
    )
    noisy = clean + rng.normal(loc=0.0, scale=noise_std, size=velocity.shape)
    return velocity, noisy, clean


def fit_spectrum_scipy(
    velocity: np.ndarray,
    transmission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit sextet parameters by bounded nonlinear least squares."""

    # [baseline, depth, isomer_shift, hyperfine_step, linewidth]
    p0 = np.array([0.98, 0.20, 0.00, 0.70, 0.24], dtype=float)
    lower = np.array([0.60, 0.01, -1.50, 0.10, 0.05], dtype=float)
    upper = np.array([1.40, 0.80, 1.50, 2.00, 1.20], dtype=float)

    popt, pcov = curve_fit(
        mossbauer_sextet_numpy,
        velocity,
        transmission,
        p0=p0,
        bounds=(lower, upper),
        maxfev=30000,
    )
    return popt, pcov


def inv_softplus(x: float) -> float:
    """Inverse softplus for positive parameter initialization in torch."""

    return math.log(math.expm1(max(x, 1e-12)))


def _decode_torch_params(raw: Any) -> Any:
    """Map unconstrained torch variables to physically valid parameters."""

    baseline = raw[0]
    depth = torch.nn.functional.softplus(raw[1])
    isomer_shift = raw[2]
    hyperfine_step = torch.nn.functional.softplus(raw[3])
    linewidth = torch.nn.functional.softplus(raw[4]) + 1e-6
    return baseline, depth, isomer_shift, hyperfine_step, linewidth


def mossbauer_sextet_torch(velocity: Any, raw_params: Any) -> Any:
    """Torch analogue of sextet model for gradient-based refinement."""

    baseline, depth, isomer_shift, hyperfine_step, linewidth = _decode_torch_params(raw_params)
    offsets_t = torch.tensor(OFFSETS, dtype=velocity.dtype, device=velocity.device)
    weights_t = torch.tensor(REL_INTENSITY, dtype=velocity.dtype, device=velocity.device)

    centers = isomer_shift + hyperfine_step * offsets_t
    profile = torch.zeros_like(velocity)
    for idx in range(6):
        profile = profile + weights_t[idx] * (linewidth**2) / ((velocity - centers[idx]) ** 2 + linewidth**2)

    return baseline - depth * profile


def refine_with_torch(
    velocity: np.ndarray,
    transmission: np.ndarray,
    init_params: np.ndarray,
    steps: int = 220,
    lr: float = 0.03,
) -> tuple[np.ndarray | None, float]:
    """Run a short torch optimization pass starting from SciPy solution."""

    if not TORCH_AVAILABLE:
        return None, math.nan

    velocity_t = torch.tensor(velocity, dtype=torch.float64)
    transmission_t = torch.tensor(transmission, dtype=torch.float64)

    raw_init = np.array(
        [
            init_params[0],
            inv_softplus(init_params[1]),
            init_params[2],
            inv_softplus(init_params[3]),
            inv_softplus(init_params[4]),
        ],
        dtype=float,
    )
    raw = torch.tensor(raw_init, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pred_t = mossbauer_sextet_torch(velocity_t, raw)
        loss = torch.mean((pred_t - transmission_t) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        baseline, depth, isomer_shift, hyperfine_step, linewidth = _decode_torch_params(raw)
        decoded = np.array(
            [
                float(baseline.item()),
                float(depth.item()),
                float(isomer_shift.item()),
                float(hyperfine_step.item()),
                float(linewidth.item()),
            ],
            dtype=float,
        )
        mse = float(torch.mean((mossbauer_sextet_torch(velocity_t, raw) - transmission_t) ** 2).item())

    return decoded, mse


def params_to_dict(values: np.ndarray) -> dict[str, float]:
    """Convert parameter vector to labeled dictionary."""

    return {
        "baseline": float(values[0]),
        "depth": float(values[1]),
        "isomer_shift_mm_per_s": float(values[2]),
        "hyperfine_step_mm_per_s": float(values[3]),
        "linewidth_mm_per_s": float(values[4]),
    }


def main() -> None:
    true_params = SextetParams(
        baseline=1.000,
        depth=0.240,
        isomer_shift=0.180,
        hyperfine_step=0.820,
        linewidth=0.210,
    )

    velocity, observed, clean = generate_synthetic_data(true_params)

    scipy_popt, _scipy_pcov = fit_spectrum_scipy(velocity, observed)
    scipy_pred = mossbauer_sextet_numpy(velocity, *scipy_popt)
    scipy_mse = mean_squared_error(observed, scipy_pred)
    scipy_r2 = r2_score(observed, scipy_pred)

    torch_popt, torch_mse = refine_with_torch(velocity, observed, scipy_popt)
    if torch_popt is not None:
        torch_pred = mossbauer_sextet_numpy(velocity, *torch_popt)
        torch_r2 = r2_score(observed, torch_pred)
    else:
        torch_pred = np.full_like(observed, np.nan)
        torch_r2 = math.nan

    rows = {
        "true": params_to_dict(
            np.array(
                [
                    true_params.baseline,
                    true_params.depth,
                    true_params.isomer_shift,
                    true_params.hyperfine_step,
                    true_params.linewidth,
                ],
                dtype=float,
            )
        ),
        "scipy_fit": params_to_dict(scipy_popt),
    }
    if torch_popt is not None:
        rows["torch_refined"] = params_to_dict(torch_popt)

    param_df = pd.DataFrame.from_dict(rows, orient="index")

    metrics_df = pd.DataFrame(
        [
            {
                "model": "scipy_fit",
                "mse": float(scipy_mse),
                "r2": float(scipy_r2),
            },
            {
                "model": "torch_refined" if torch_popt is not None else "torch_refined(skipped)",
                "mse": float(torch_mse),
                "r2": float(torch_r2),
            },
        ]
    )

    sample_slice = slice(None, None, 40)
    sample_df = pd.DataFrame(
        {
            "velocity_mm_per_s": velocity[sample_slice],
            "observed": observed[sample_slice],
            "clean_truth": clean[sample_slice],
            "scipy_fit": scipy_pred[sample_slice],
            "torch_refined": torch_pred[sample_slice],
        }
    )

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x: .6f}")

    print("Mössbauer Spectroscopy MVP (Fe-57-like sextet fit)")
    print("=" * 72)
    print("[Parameter Comparison]")
    print(param_df)
    print()
    print("[Fit Metrics]")
    print(metrics_df)
    print()
    print("[Sampled Spectrum Rows]")
    print(sample_df)

    if not TORCH_AVAILABLE:
        print()
        print("Note: PyTorch is not available in this runtime; torch refinement was skipped.")


if __name__ == "__main__":
    main()
