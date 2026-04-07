"""Minimal runnable MVP for Liquid Drop Model (LDM)."""

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
class LDMParams:
    """Core coefficient set for the liquid drop model."""

    a_v: float
    a_s: float
    a_c: float
    a_a: float
    a_p: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a_v, self.a_s, self.a_c, self.a_a, self.a_p], dtype=np.float64)

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "LDMParams":
        v = np.asarray(list(values), dtype=np.float64)
        if v.shape != (5,):
            raise ValueError("LDM coefficient array must have length 5")
        return cls(
            a_v=float(v[0]),
            a_s=float(v[1]),
            a_c=float(v[2]),
            a_a=float(v[3]),
            a_p=float(v[4]),
        )


TEXTBOOK_PARAMS = LDMParams(a_v=15.8, a_s=18.3, a_c=0.714, a_a=23.2, a_p=12.0)


def pairing_sign(a: int, z: int) -> int:
    """Return +1 for even-even, -1 for odd-odd, 0 for odd-A nuclei."""
    if a % 2 == 1:
        return 0
    n = a - z
    if (z % 2 == 0) and (n % 2 == 0):
        return 1
    if (z % 2 == 1) and (n % 2 == 1):
        return -1
    return 0


def design_matrix(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Build explicit linear features for LDM energy terms."""
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


def liquid_drop_binding_energy(a: int, z: int, params: LDMParams) -> float:
    """Compute total nuclear binding energy B(A, Z) in MeV."""
    aa = float(a)
    zz = float(z)
    pair = float(pairing_sign(a, z))
    return (
        params.a_v * aa
        - params.a_s * (aa ** (2.0 / 3.0))
        - params.a_c * zz * (zz - 1.0) / (aa ** (1.0 / 3.0))
        - params.a_a * ((aa - 2.0 * zz) ** 2.0) / aa
        + params.a_p * pair / np.sqrt(aa)
    )


def build_reference_dataset() -> pd.DataFrame:
    """Representative nuclei with total binding energies (MeV)."""
    rows: List[Tuple[str, int, int, float]] = [
        ("2H", 2, 1, 2.2246),
        ("4He", 4, 2, 28.2957),
        ("12C", 12, 6, 92.1620),
        ("16O", 16, 8, 127.6200),
        ("20Ne", 20, 10, 160.6450),
        ("40Ca", 40, 20, 342.0520),
        ("48Ca", 48, 20, 415.9900),
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


def fit_params_scipy(df: pd.DataFrame, init: LDMParams) -> LDMParams:
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

    return LDMParams.from_array(result.x)


def fit_params_linear(df: pd.DataFrame) -> LDMParams:
    x = design_matrix(df["A"].to_numpy(), df["Z"].to_numpy())
    y = df["B_exp_MeV"].to_numpy(dtype=np.float64)

    if SKLEARN_AVAILABLE:
        model = LinearRegression(fit_intercept=False)
        model.fit(x, y)
        coeffs = model.coef_
    else:
        coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    return LDMParams.from_array(coeffs)


def evaluate_dataset(df: pd.DataFrame, params: LDMParams) -> Tuple[pd.DataFrame, Dict[str, float]]:
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


def fissility_parameter(a: int, z: int, params: LDMParams) -> float:
    """x = E_c0 / (2 E_s0), where x >= 1 implies vanishing LDM fission barrier."""
    aa = float(a)
    zz = float(z)
    e_surface = params.a_s * (aa ** (2.0 / 3.0))
    e_coulomb = params.a_c * zz * (zz - 1.0) / (aa ** (1.0 / 3.0))
    return float(e_coulomb / (2.0 * e_surface))


def surface_curvature_at_sphere(a: int, z: int, params: LDMParams) -> float:
    """Quadratic curvature coefficient around spherical shape."""
    aa = float(a)
    zz = float(z)
    e_surface = params.a_s * (aa ** (2.0 / 3.0))
    e_coulomb = params.a_c * zz * (zz - 1.0) / (aa ** (1.0 / 3.0))
    return float((2.0 * e_surface - e_coulomb) / 5.0)


def fission_barrier_mev(a: int, z: int, params: LDMParams) -> float:
    """Empirical LDM barrier estimate for x < 1; clipped to zero for x >= 1."""
    x = fissility_parameter(a, z, params)
    if x >= 1.0:
        return 0.0
    return float(0.36 * params.a_s * (float(a) ** (2.0 / 3.0)) * ((1.0 - x) ** 2.0))


def deformation_energy_curve(
    a: int,
    z: int,
    params: LDMParams,
    alpha_max: float = 1.8,
    num_points: int = 181,
    k4: float = 0.18,
) -> pd.DataFrame:
    """Landau-like deformation curve: DeltaE = Es[(1-x) a^2 + k4 a^4]."""
    if num_points < 5:
        raise ValueError("num_points must be >= 5")

    aa = float(a)
    e_surface = params.a_s * (aa ** (2.0 / 3.0))
    x = fissility_parameter(a, z, params)

    alpha = np.linspace(0.0, alpha_max, num_points)
    delta_e = e_surface * ((1.0 - x) * (alpha**2) + k4 * (alpha**4))

    out = pd.DataFrame({"alpha": alpha, "deltaE_MeV": delta_e})
    out["A"] = int(a)
    out["Z"] = int(z)
    return out


def classify_by_fissility(x: float) -> str:
    if x < 0.70:
        return "low_fissility"
    if x < 1.00:
        return "metastable_fissionable"
    return "barrierless_in_ldm"


def build_fissility_table(params: LDMParams) -> pd.DataFrame:
    nuclei: List[Tuple[str, int, int]] = [
        ("90Zr", 90, 40),
        ("120Sn", 120, 50),
        ("208Pb", 208, 82),
        ("232Th", 232, 90),
        ("238U", 238, 92),
        ("252Cf", 252, 98),
        ("X280_170", 280, 170),
    ]

    rows: List[Tuple[str, int, int, float, float, float, str]] = []
    for name, a, z in nuclei:
        x = fissility_parameter(a, z, params)
        barrier = fission_barrier_mev(a, z, params)
        curvature = surface_curvature_at_sphere(a, z, params)
        rows.append((name, a, z, x, barrier, curvature, classify_by_fissility(x)))

    return pd.DataFrame(
        rows,
        columns=["nuclide", "A", "Z", "fissility_x", "barrier_MeV", "curvature_MeV", "class"],
    )


def stable_z_continuous(a: int, params: LDMParams) -> float:
    """Continuous beta-stability estimate from dB/dZ = 0 (without pairing term)."""
    aa = float(a)
    numerator = 4.0 * params.a_a * aa + params.a_c * (aa ** (2.0 / 3.0))
    denominator = 8.0 * params.a_a + 2.0 * params.a_c * (aa ** (2.0 / 3.0))
    return float(numerator / denominator)


def best_integer_z_for_a(a: int, params: LDMParams) -> Tuple[int, float, float]:
    z_cont = stable_z_continuous(a, params)
    center = int(round(z_cont))
    lo = max(1, center - 6)
    hi = min(a - 1, center + 6)

    best_z = lo
    best_bea = -np.inf
    for z in range(lo, hi + 1):
        be = liquid_drop_binding_energy(a, z, params)
        bea = be / float(a)
        if bea > best_bea:
            best_bea = bea
            best_z = z

    return best_z, z_cont, float(best_bea)


def scan_binding_peak(params: LDMParams, a_min: int = 20, a_max: int = 260) -> pd.DataFrame:
    rows: List[Tuple[int, int, float, float]] = []
    for a in range(a_min, a_max + 1):
        z_best, z_cont, bea = best_integer_z_for_a(a, params)
        x = fissility_parameter(a, z_best, params)
        rows.append((a, z_best, z_cont, bea, x))

    return pd.DataFrame(rows, columns=["A", "Z_best", "Z_continuous", "BEA_max", "fissility_x"])


def torch_cross_check(df: pd.DataFrame, params: LDMParams) -> float:
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
    fitted_linear = fit_params_linear(df)

    report_df, metrics = evaluate_dataset(df, fitted_scipy)

    coeff_table = pd.DataFrame(
        {
            "coef": ["a_v", "a_s", "a_c", "a_a", "a_p"],
            "textbook": TEXTBOOK_PARAMS.as_array(),
            "scipy_fit": fitted_scipy.as_array(),
            "linear_fit": fitted_linear.as_array(),
        }
    )
    coeff_table["abs_diff_scipy_linear"] = np.abs(coeff_table["scipy_fit"] - coeff_table["linear_fit"])

    fiss_df = build_fissility_table(fitted_scipy)

    peak_df = scan_binding_peak(fitted_scipy, a_min=20, a_max=260)
    peak_row = peak_df.iloc[int(peak_df["BEA_max"].idxmax())]
    peak_a = int(peak_row["A"])
    peak_z = int(peak_row["Z_best"])
    peak_bea = float(peak_row["BEA_max"])

    deform_u = deformation_energy_curve(238, 92, fitted_scipy)
    deform_x = deformation_energy_curve(280, 170, fitted_scipy)
    u_min = deform_u.iloc[int(deform_u["deltaE_MeV"].idxmin())]
    x_min = deform_x.iloc[int(deform_x["deltaE_MeV"].idxmin())]

    torch_max_diff = torch_cross_check(df, fitted_scipy)

    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 20)

    print("=== Liquid Drop Model (LDM) MVP ===")
    print("\n[1] Fitted coefficients")
    print(coeff_table.to_string(index=False, float_format=lambda v: f"{v:10.6f}"))

    print("\n[2] Dataset prediction report")
    print(
        report_df[["nuclide", "A", "Z", "B_exp_MeV", "B_pred_MeV", "error_MeV", "BEA_pred"]].to_string(
            index=False,
            float_format=lambda v: f"{v:10.4f}",
        )
    )

    print("\n[3] Error metrics")
    print(f"MAE      : {metrics['mae_mev']:.4f} MeV")
    print(f"RMSE     : {metrics['rmse_mev']:.4f} MeV")
    print(f"Max |err|: {metrics['max_abs_mev']:.4f} MeV")
    print(f"R^2      : {metrics['r2']:.6f}")

    print("\n[4] Fissility and barrier summary")
    print(fiss_df.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    print("\n[5] Predicted binding-energy-per-nucleon peak")
    print(f"A_peak={peak_a}, Z_peak={peak_z}, max(B/A)={peak_bea:.4f} MeV")

    print("\n[6] Deformation curve minima")
    print(
        "U-238: "
        f"alpha_min={float(u_min['alpha']):.3f}, "
        f"deltaE_min={float(u_min['deltaE_MeV']):.3f} MeV"
    )
    print(
        "X280_170: "
        f"alpha_min={float(x_min['alpha']):.3f}, "
        f"deltaE_min={float(x_min['deltaE_MeV']):.3f} MeV"
    )

    if np.isfinite(torch_max_diff):
        print(f"\n[7] Torch cross-check max |delta| = {torch_max_diff:.3e}")
    else:
        print("\n[7] Torch cross-check skipped (torch not available)")

    max_coeff_gap = float(coeff_table["abs_diff_scipy_linear"].max())

    u_row = fiss_df[fiss_df["nuclide"] == "238U"].iloc[0]
    x_row = fiss_df[fiss_df["nuclide"] == "X280_170"].iloc[0]

    assert metrics["mae_mev"] < 20.0, "MAE is too large; coefficient fit may be broken"
    assert metrics["r2"] > 0.995, "R^2 is unexpectedly low"
    assert max_coeff_gap < 2.0, "SciPy and linear fits diverge too much"
    assert 45 <= peak_a <= 75, "B/A peak should appear around iron region"

    assert 0.55 <= float(u_row["fissility_x"]) <= 0.85, "U-238 fissility should be in expected range"
    assert float(u_row["barrier_MeV"]) > 0.0, "U-238 should keep a positive LDM barrier"

    assert float(x_row["fissility_x"]) > 1.0, "X280_170 should be beyond x=1"
    assert float(x_row["barrier_MeV"]) == 0.0, "Barrier must vanish for x>=1 in this model"
    assert float(x_min["deltaE_MeV"]) < 0.0, "x>1 deformation curve should develop a lower-energy deformed minimum"

    if np.isfinite(torch_max_diff):
        assert torch_max_diff < 1e-10, "Torch and NumPy predictions diverged"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
