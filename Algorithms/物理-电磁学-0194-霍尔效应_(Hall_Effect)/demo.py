"""Minimal runnable MVP for Hall Effect parameter estimation.

This script demonstrates a practical Hall-effect workflow:
1) Build deterministic synthetic measurements under +B/-B sweeps.
2) Use antisymmetrization to remove offset/even-in-B parasitic voltage.
3) Estimate Hall coefficient R_H with explicit least squares.
4) Infer carrier type and carrier density from R_H sign and magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

E_CHARGE = 1.602_176_634e-19  # Coulomb


@dataclass(frozen=True)
class HallCaseConfig:
    """Configuration for one synthetic Hall experiment case."""

    name: str
    seed: int
    true_r_h: float  # m^3/C, sign indicates carrier type
    thickness: float  # m
    n_points: int
    offset_uV: float
    even_coeff_uV_per_T2: float
    noise_uV: float


def check_inputs(current: np.ndarray, b_field: np.ndarray, v_plus: np.ndarray, v_minus: np.ndarray, thickness: float) -> None:
    """Validate shapes and numerical sanity."""
    arrays = {
        "current": current,
        "b_field": b_field,
        "v_plus": v_plus,
        "v_minus": v_minus,
    }
    for name, arr in arrays.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape={arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")

    n = current.size
    if b_field.size != n or v_plus.size != n or v_minus.size != n:
        raise ValueError("current, b_field, v_plus, v_minus must have the same length.")
    if n < 3:
        raise ValueError("At least 3 samples are required for a stable fit.")
    if thickness <= 0.0 or not np.isfinite(thickness):
        raise ValueError("thickness must be finite and > 0.")


def antisymmetrize_hall_voltage(v_plus: np.ndarray, v_minus: np.ndarray) -> np.ndarray:
    """Extract odd-in-B Hall component: V_H = (V(+B) - V(-B)) / 2."""
    return 0.5 * (v_plus - v_minus)


def estimate_hall_coefficient(
    current: np.ndarray,
    b_field: np.ndarray,
    v_plus: np.ndarray,
    v_minus: np.ndarray,
    thickness: float,
) -> Dict[str, float]:
    """Estimate Hall coefficient by explicit least squares with intercept.

    Model after antisymmetrization:
        y = R_H * x + c
    where:
        y = V_H,
        x = I * B / t,
        c ideally ~ 0 (residual odd parasitic term).
    """
    check_inputs(current, b_field, v_plus, v_minus, thickness)

    y = antisymmetrize_hall_voltage(v_plus, v_minus)
    x = (current * b_field) / thickness

    a = np.column_stack([x, np.ones_like(x)])
    beta, residuals, rank, _ = np.linalg.lstsq(a, y, rcond=None)
    if rank < 2:
        raise RuntimeError("Linear system is rank-deficient; cannot estimate R_H reliably.")

    r_h_hat = float(beta[0])
    intercept = float(beta[1])

    y_hat = a @ beta
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 1.0

    return {
        "r_h_hat": r_h_hat,
        "intercept": intercept,
        "r2": r2,
        "sse": sse,
        "n_samples": float(current.size),
    }


def infer_carrier_properties(r_h: float) -> Dict[str, float | str]:
    """Infer carrier type and density from Hall coefficient."""
    if not np.isfinite(r_h) or r_h == 0.0:
        raise ValueError("r_h must be finite and non-zero.")

    carrier_type = "electrons (n-type)" if r_h < 0.0 else "holes (p-type)"
    density = 1.0 / (abs(r_h) * E_CHARGE)  # 1/m^3
    return {
        "carrier_type": carrier_type,
        "carrier_density_m^-3": float(density),
    }


def make_synthetic_case(config: HallCaseConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic +B/-B Hall measurements with realistic parasitics.

    Measurement model:
        V_plus  = V_offset + V_even(B) + V_hall + noise
        V_minus = V_offset + V_even(B) - V_hall + noise
    with
        V_hall = R_H * I * B / t
    """
    rng = np.random.default_rng(config.seed)

    current = rng.uniform(0.005, 0.040, size=config.n_points)  # A
    b_field = rng.uniform(0.15, 1.20, size=config.n_points)  # T

    v_hall = config.true_r_h * current * b_field / config.thickness
    v_offset = config.offset_uV * 1e-6
    v_even = (config.even_coeff_uV_per_T2 * 1e-6) * (b_field**2)

    noise_plus = rng.normal(0.0, config.noise_uV * 1e-6, size=config.n_points)
    noise_minus = rng.normal(0.0, config.noise_uV * 1e-6, size=config.n_points)

    v_plus = v_offset + v_even + v_hall + noise_plus
    v_minus = v_offset + v_even - v_hall + noise_minus
    return current, b_field, v_plus, v_minus


def run_case(config: HallCaseConfig) -> Dict[str, float | str]:
    """Run one synthetic Hall case and return evaluation metrics."""
    current, b_field, v_plus, v_minus = make_synthetic_case(config)

    fit = estimate_hall_coefficient(
        current=current,
        b_field=b_field,
        v_plus=v_plus,
        v_minus=v_minus,
        thickness=config.thickness,
    )
    inferred = infer_carrier_properties(float(fit["r_h_hat"]))

    true_density = 1.0 / (abs(config.true_r_h) * E_CHARGE)
    abs_err = abs(float(fit["r_h_hat"]) - config.true_r_h)
    rel_err = abs_err / abs(config.true_r_h)
    density_rel_err = abs(float(inferred["carrier_density_m^-3"]) - true_density) / true_density

    report: Dict[str, float | str] = {
        "name": config.name,
        "true_r_h": config.true_r_h,
        "estimated_r_h": float(fit["r_h_hat"]),
        "abs_error_r_h": abs_err,
        "rel_error_r_h": rel_err,
        "intercept_V": float(fit["intercept"]),
        "r2": float(fit["r2"]),
        "carrier_type": str(inferred["carrier_type"]),
        "true_density_m^-3": true_density,
        "estimated_density_m^-3": float(inferred["carrier_density_m^-3"]),
        "density_rel_error": density_rel_err,
        "n_samples": float(fit["n_samples"]),
    }
    return report


def print_report(report: Dict[str, float | str]) -> None:
    """Pretty print one case report."""
    print(f"\n=== Case: {report['name']} ===")
    print("R_H true      : {:.6e} m^3/C".format(float(report["true_r_h"])))
    print("R_H estimated : {:.6e} m^3/C".format(float(report["estimated_r_h"])))
    print("R_H abs error : {:.6e}".format(float(report["abs_error_r_h"])))
    print("R_H rel error : {:.6e}".format(float(report["rel_error_r_h"])))
    print("fit intercept : {:.6e} V".format(float(report["intercept_V"])))
    print("fit R^2       : {:.6f}".format(float(report["r2"])))
    print("carrier type  : {}".format(str(report["carrier_type"])))
    print("density true  : {:.6e} 1/m^3".format(float(report["true_density_m^-3"])))
    print("density est   : {:.6e} 1/m^3".format(float(report["estimated_density_m^-3"])))
    print("density rel err: {:.6e}".format(float(report["density_rel_error"])))
    print("samples       : {:.0f}".format(float(report["n_samples"])))


def main() -> None:
    cases = [
        HallCaseConfig(
            name="N-type semiconductor",
            seed=7,
            true_r_h=-6.5e-4,
            thickness=500e-6,
            n_points=80,
            offset_uV=18.0,
            even_coeff_uV_per_T2=2.8,
            noise_uV=1.2,
        ),
        HallCaseConfig(
            name="P-type semiconductor",
            seed=19,
            true_r_h=4.2e-4,
            thickness=350e-6,
            n_points=80,
            offset_uV=-12.0,
            even_coeff_uV_per_T2=3.5,
            noise_uV=1.0,
        ),
    ]

    reports = [run_case(cfg) for cfg in cases]
    for report in reports:
        print_report(report)

    max_rel_error = max(float(r["rel_error_r_h"]) for r in reports)
    max_density_rel_error = max(float(r["density_rel_error"]) for r in reports)
    min_r2 = min(float(r["r2"]) for r in reports)
    min_abs_intercept = max(abs(float(r["intercept_V"])) for r in reports)

    pass_flag = (
        max_rel_error < 0.03
        and max_density_rel_error < 0.03
        and min_r2 > 0.999
        and min_abs_intercept < 8e-6
    )

    print("\n=== Summary ===")
    print(f"max R_H relative error      : {max_rel_error:.6e}")
    print(f"max density relative error  : {max_density_rel_error:.6e}")
    print(f"min fit R^2                 : {min_r2:.6f}")
    print(f"max |fit intercept|         : {min_abs_intercept:.6e} V")
    print(f"all checks pass             : {pass_flag}")

    assert pass_flag, "Hall-effect MVP quality gate failed."


if __name__ == "__main__":
    main()
