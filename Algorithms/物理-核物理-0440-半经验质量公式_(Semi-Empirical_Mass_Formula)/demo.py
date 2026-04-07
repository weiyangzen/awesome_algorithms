"""Minimal runnable MVP for Semi-Empirical Mass Formula (SEMF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class SEMFParams:
    """Coefficients in the Weizsaecker semi-empirical mass formula."""

    a_v: float
    a_s: float
    a_c: float
    a_a: float
    a_p: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a_v, self.a_s, self.a_c, self.a_a, self.a_p], dtype=np.float64)

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "SEMFParams":
        v = np.asarray(list(values), dtype=np.float64)
        if v.shape != (5,):
            raise ValueError("SEMF parameter array must have length 5")
        return cls(a_v=float(v[0]), a_s=float(v[1]), a_c=float(v[2]), a_a=float(v[3]), a_p=float(v[4]))


TEXTBOOK_PARAMS = SEMFParams(a_v=15.8, a_s=18.3, a_c=0.714, a_a=23.2, a_p=12.0)


def pairing_sign(a: int, z: int) -> int:
    """Return +1 (even-even), -1 (odd-odd), 0 (odd A)."""
    if a % 2 == 1:
        return 0
    n = a - z
    z_even = (z % 2) == 0
    n_even = (n % 2) == 0
    if z_even and n_even:
        return 1
    if (not z_even) and (not n_even):
        return -1
    return 0


def design_matrix(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Construct explicit SEMF basis features for linear-in-parameter fitting."""
    a = np.asarray(a, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if a.shape != z.shape:
        raise ValueError("A and Z must have the same shape")

    pair = np.array([pairing_sign(int(ai), int(zi)) for ai, zi in zip(a, z)], dtype=np.float64)
    return np.column_stack(
        [
            a,
            -(a ** (2.0 / 3.0)),
            -(z * (z - 1.0) / (a ** (1.0 / 3.0))),
            -(((a - 2.0 * z) ** 2.0) / a),
            pair / np.sqrt(a),
        ]
    )


def semf_binding_energy(a: int, z: int, params: SEMFParams) -> float:
    """Compute total binding energy B(A, Z) in MeV."""
    aa = float(a)
    zz = float(z)
    pair = pairing_sign(a, z)
    return (
        params.a_v * aa
        - params.a_s * (aa ** (2.0 / 3.0))
        - params.a_c * zz * (zz - 1.0) / (aa ** (1.0 / 3.0))
        - params.a_a * ((aa - 2.0 * zz) ** 2.0) / aa
        + params.a_p * pair / np.sqrt(aa)
    )


def build_reference_dataset() -> pd.DataFrame:
    """Representative reference points (total binding energies, MeV)."""
    rows: List[Tuple[str, int, int, float]] = [
        ("2H", 2, 1, 2.2246),
        ("4He", 4, 2, 28.2957),
        ("12C", 12, 6, 92.1620),
        ("16O", 16, 8, 127.6200),
        ("20Ne", 20, 10, 160.6450),
        ("40Ca", 40, 20, 342.0520),
        ("56Fe", 56, 26, 492.2540),
        ("58Ni", 58, 28, 506.4600),
        ("62Ni", 62, 28, 545.2600),
        ("90Zr", 90, 40, 783.8900),
        ("120Sn", 120, 50, 1020.5500),
        ("132Sn", 132, 50, 1102.9000),
        ("208Pb", 208, 82, 1636.4300),
        ("238U", 238, 92, 1801.6900),
    ]
    df = pd.DataFrame(rows, columns=["nuclide", "A", "Z", "B_exp_MeV"])
    df["N"] = df["A"] - df["Z"]
    return df


def fit_params_scipy(df: pd.DataFrame, init: SEMFParams) -> SEMFParams:
    x = design_matrix(df["A"].to_numpy(), df["Z"].to_numpy())
    y = df["B_exp_MeV"].to_numpy(dtype=np.float64)

    def residuals(coeffs: np.ndarray) -> np.ndarray:
        return x @ coeffs - y

    bounds_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    bounds_high = np.array([40.0, 40.0, 5.0, 60.0, 40.0], dtype=np.float64)

    result = least_squares(
        residuals,
        x0=init.as_array(),
        bounds=(bounds_low, bounds_high),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"SciPy least_squares failed: {result.message}")

    return SEMFParams.from_array(result.x)


def fit_params_linear_regression(df: pd.DataFrame) -> SEMFParams:
    x = design_matrix(df["A"].to_numpy(), df["Z"].to_numpy())
    y = df["B_exp_MeV"].to_numpy(dtype=np.float64)

    if SKLEARN_AVAILABLE:
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        coeffs = model.coef_
    else:
        coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    return SEMFParams.from_array(coeffs)


def evaluate_on_dataset(df: pd.DataFrame, params: SEMFParams) -> Tuple[pd.DataFrame, Dict[str, float]]:
    x = design_matrix(df["A"].to_numpy(), df["Z"].to_numpy())
    y = df["B_exp_MeV"].to_numpy(dtype=np.float64)
    pred = x @ params.as_array()
    err = pred - y

    out = df.copy()
    out["B_pred_MeV"] = pred
    out["error_MeV"] = err
    out["BEA_pred"] = out["B_pred_MeV"] / out["A"]

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(err**2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    metrics = {
        "mae_mev": mae,
        "rmse_mev": rmse,
        "max_abs_mev": max_abs,
        "r2": float(r2),
    }
    return out, metrics


def stable_z_continuous(a: int, params: SEMFParams) -> float:
    aa = float(a)
    numerator = 4.0 * params.a_a * aa + params.a_c * (aa ** (2.0 / 3.0))
    denominator = 8.0 * params.a_a + 2.0 * params.a_c * (aa ** (2.0 / 3.0))
    return numerator / denominator


def best_integer_z_for_a(a: int, params: SEMFParams) -> Tuple[int, float, float]:
    z_star = stable_z_continuous(a, params)
    center = int(round(z_star))
    lo = max(1, center - 4)
    hi = min(a - 1, center + 4)

    best_z = lo
    best_bea = -np.inf
    for z in range(lo, hi + 1):
        be = semf_binding_energy(a, z, params)
        bea = be / float(a)
        if bea > best_bea:
            best_bea = bea
            best_z = z

    return best_z, z_star, best_bea


def scan_iron_peak(params: SEMFParams, a_min: int = 20, a_max: int = 260) -> pd.DataFrame:
    rows: List[Tuple[int, int, float, float]] = []
    for a in range(a_min, a_max + 1):
        z_best, z_cont, bea = best_integer_z_for_a(a, params)
        rows.append((a, z_best, z_cont, bea))

    df = pd.DataFrame(rows, columns=["A", "Z_best", "Z_continuous", "BEA_max"])
    return df


def torch_cross_check(df: pd.DataFrame, params: SEMFParams) -> float:
    if not TORCH_AVAILABLE:
        return float("nan")

    a = torch.tensor(df["A"].to_numpy(), dtype=torch.float64)
    z = torch.tensor(df["Z"].to_numpy(), dtype=torch.float64)
    pair = torch.tensor(
        [pairing_sign(int(ai), int(zi)) for ai, zi in zip(df["A"].to_numpy(), df["Z"].to_numpy())],
        dtype=torch.float64,
    )

    x_t = torch.stack(
        [
            a,
            -(a ** (2.0 / 3.0)),
            -(z * (z - 1.0) / (a ** (1.0 / 3.0))),
            -(((a - 2.0 * z) ** 2.0) / a),
            pair / torch.sqrt(a),
        ],
        dim=1,
    )
    coeffs = torch.tensor(params.as_array(), dtype=torch.float64)
    pred_t = (x_t @ coeffs).detach().cpu().numpy()

    pred_np = design_matrix(df["A"].to_numpy(), df["Z"].to_numpy()) @ params.as_array()
    return float(np.max(np.abs(pred_t - pred_np)))


def main() -> None:
    df = build_reference_dataset()

    fitted_scipy = fit_params_scipy(df, TEXTBOOK_PARAMS)
    fitted_lr = fit_params_linear_regression(df)

    report_df, metrics = evaluate_on_dataset(df, fitted_scipy)

    coeff_table = pd.DataFrame(
        {
            "coef": ["a_v", "a_s", "a_c", "a_a", "a_p"],
            "textbook": TEXTBOOK_PARAMS.as_array(),
            "scipy_fit": fitted_scipy.as_array(),
            "linear_fit": fitted_lr.as_array(),
        }
    )
    coeff_table["abs_diff_scipy_linear"] = np.abs(coeff_table["scipy_fit"] - coeff_table["linear_fit"])

    valley_rows: List[Tuple[int, float, int, int]] = []
    for a in [20, 40, 56, 90, 120, 132, 208, 238]:
        z_best, z_cont, _ = best_integer_z_for_a(a, fitted_scipy)
        valley_rows.append((a, z_cont, z_best, a - z_best))
    valley_df = pd.DataFrame(valley_rows, columns=["A", "Z_continuous", "Z_best_integer", "N_best_integer"])

    peak_df = scan_iron_peak(fitted_scipy, a_min=20, a_max=260)
    peak_idx = int(peak_df["BEA_max"].idxmax())
    peak_row = peak_df.iloc[peak_idx]
    peak_a = int(peak_row["A"])
    peak_z = int(peak_row["Z_best"])
    peak_bea = float(peak_row["BEA_max"])

    torch_max_diff = torch_cross_check(df, fitted_scipy)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=== Semi-Empirical Mass Formula (SEMF) MVP ===")
    print("\n[1] Fitted coefficients")
    print(coeff_table.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    print("\n[2] Dataset prediction report")
    print(
        report_df[["nuclide", "A", "Z", "B_exp_MeV", "B_pred_MeV", "error_MeV", "BEA_pred"]].to_string(
            index=False, float_format=lambda x: f"{x:10.4f}"
        )
    )

    print("\n[3] Error metrics (SciPy fit)")
    print(f"MAE      : {metrics['mae_mev']:.4f} MeV")
    print(f"RMSE     : {metrics['rmse_mev']:.4f} MeV")
    print(f"Max |err|: {metrics['max_abs_mev']:.4f} MeV")
    print(f"R^2      : {metrics['r2']:.6f}")

    print("\n[4] Beta-stability valley samples")
    print(valley_df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    print("\n[5] Predicted iron peak from SEMF scan")
    print(f"A_peak={peak_a}, Z_peak={peak_z}, max(B/A)={peak_bea:.4f} MeV")

    if np.isfinite(torch_max_diff):
        print(f"\n[6] Torch cross-check max |delta| = {torch_max_diff:.3e}")
    else:
        print("\n[6] Torch cross-check skipped (torch not available)")

    max_coeff_gap = float(coeff_table["abs_diff_scipy_linear"].max())

    assert metrics["mae_mev"] < 20.0, "MAE is too large; fit may be broken"
    assert metrics["r2"] > 0.995, "R^2 is unexpectedly low"
    assert max_coeff_gap < 2.0, "SciPy and linear-regression fits disagree too much"
    assert 45 <= peak_a <= 75, "Iron peak should appear around A~56"
    if np.isfinite(torch_max_diff):
        assert torch_max_diff < 1e-10, "Torch and NumPy predictions diverged"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
