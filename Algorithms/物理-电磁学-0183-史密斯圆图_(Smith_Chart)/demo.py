"""Minimal runnable MVP for Smith Chart based matching.

This script provides an explicit, source-level implementation of core
Smith-chart computations instead of relying on black-box RF packages:
- impedance <-> reflection coefficient mapping,
- movement along a lossless transmission line,
- intersection search with r=1 circle,
- series-reactance synthesis to reach exact match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class SmithMatchCase:
    """Configuration of one impedance-matching demonstration case."""

    name: str
    z0_ohm: float
    z_load_ohm: complex
    freq_hz: float
    grid_points: int = 6001


def _is_finite_complex(z: complex) -> bool:
    return bool(np.isfinite(z.real) and np.isfinite(z.imag))


def normalize_impedance(z_load_ohm: complex, z0_ohm: float) -> complex:
    """Normalize load impedance by characteristic impedance."""
    if z0_ohm <= 0.0 or not np.isfinite(z0_ohm):
        raise ValueError("z0_ohm must be finite and > 0.")
    if not _is_finite_complex(z_load_ohm):
        raise ValueError("z_load_ohm must be finite complex.")
    return z_load_ohm / z0_ohm


def gamma_from_z_norm(z_norm: complex) -> complex:
    """Map normalized impedance z to reflection coefficient Gamma."""
    den = z_norm + 1.0
    if abs(den) < EPS:
        raise ValueError("z_norm is too close to -1, Gamma mapping is singular.")
    return (z_norm - 1.0) / den


def z_norm_from_gamma(gamma: complex) -> complex:
    """Inverse map from reflection coefficient Gamma to normalized impedance z."""
    den = 1.0 - gamma
    if abs(den) < EPS:
        raise ValueError("gamma is too close to 1, impedance mapping is singular.")
    return (1.0 + gamma) / den


def vswr_from_gamma(gamma: complex) -> float:
    """Compute VSWR from reflection coefficient."""
    rho = abs(gamma)
    if rho >= 1.0:
        return float("inf")
    return float((1.0 + rho) / (1.0 - rho))


def input_impedance_norm(z_load_norm: complex, theta: float) -> complex:
    """Input normalized impedance after moving theta=beta*l toward generator.

    Formula for lossless line:
        z_in = (z_L + j tan(theta)) / (1 + j z_L tan(theta))
    """
    t = np.tan(theta)
    den = 1.0 + 1j * z_load_norm * t
    if abs(den) < EPS:
        raise RuntimeError("Transmission-line transform denominator is near zero.")
    return (z_load_norm + 1j * t) / den


def gamma_after_line(gamma_load: complex, theta: float) -> complex:
    """Rotate Gamma by lossless line movement: Gamma_in = Gamma_L * exp(-j2theta)."""
    return gamma_load * np.exp(-2j * theta)


def smith_circle_residuals(z_norm: complex) -> Tuple[float, float]:
    """Return residuals to constant-r and constant-x circle equations.

    r-circle in Gamma-plane:
        (u - r/(1+r))^2 + v^2 = (1/(1+r))^2
    x-circle in Gamma-plane (x != 0):
        (u - 1)^2 + (v - 1/x)^2 = (1/x)^2
    """
    r = float(np.real(z_norm))
    x = float(np.imag(z_norm))
    if abs(1.0 + r) < EPS:
        raise ValueError("r too close to -1, r-circle equation becomes singular.")

    gamma = gamma_from_z_norm(z_norm)
    u = float(np.real(gamma))
    v = float(np.imag(gamma))

    center_r = r / (1.0 + r)
    radius_r = 1.0 / (1.0 + r)
    res_r = (u - center_r) ** 2 + v**2 - radius_r**2

    if abs(x) < 1e-10:
        res_x = 0.0
    else:
        center_y = 1.0 / x
        radius_x = abs(1.0 / x)
        res_x = (u - 1.0) ** 2 + (v - center_y) ** 2 - radius_x**2
    return float(res_r), float(res_x)


def _bisect_root(fn, left: float, right: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """Simple bisection root finder for scalar continuous function."""
    f_left = float(fn(left))
    f_right = float(fn(right))

    if abs(f_left) < tol:
        return left
    if abs(f_right) < tol:
        return right
    if f_left * f_right > 0.0:
        raise ValueError("Bisection requires opposite signs on interval endpoints.")

    a, b = left, right
    fa, fb = f_left, f_right
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = float(fn(mid))
        if abs(fm) < tol or (b - a) * 0.5 < tol:
            return mid
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def find_r_one_intersections(z_load_norm: complex, grid_points: int = 6001) -> List[float]:
    """Find theta in [0, pi] such that Re(z_in(theta)) = 1.

    On Smith chart this corresponds to intersections of the constant-|Gamma|
    circle with the r=1 circle. Typical passive unmatched loads yield two roots
    in one half-wavelength period.
    """
    if grid_points < 101:
        raise ValueError("grid_points must be >= 101.")

    def target(theta: float) -> float:
        return float(np.real(input_impedance_norm(z_load_norm, theta)) - 1.0)

    thetas = np.linspace(0.0, np.pi, grid_points)
    vals = np.array([target(t) for t in thetas])

    roots: List[float] = []
    tol_hit = 1e-10

    for i in range(grid_points - 1):
        a, b = float(thetas[i]), float(thetas[i + 1])
        fa, fb = float(vals[i]), float(vals[i + 1])

        if abs(fa) < tol_hit:
            roots.append(a)

        if fa * fb < 0.0:
            roots.append(_bisect_root(target, a, b))
        elif abs(fb) < tol_hit:
            roots.append(b)

    roots_sorted = sorted(roots)
    deduped: List[float] = []
    for root in roots_sorted:
        if not deduped or abs(root - deduped[-1]) > 1e-6:
            deduped.append(root)

    return deduped


def series_reactance_for_match(z_in_norm: complex) -> float:
    """Return normalized series reactance x_s to force z=1+j0.

    After line section gives z_in = 1 + jx, adding series reactance -jx matches.
    """
    return float(-np.imag(z_in_norm))


def reactance_to_component(x_ohm: float, freq_hz: float) -> Tuple[str, float]:
    """Convert reactance at given frequency to ideal L/C value."""
    if not np.isfinite(x_ohm):
        raise ValueError("x_ohm must be finite.")
    if freq_hz <= 0.0 or not np.isfinite(freq_hz):
        raise ValueError("freq_hz must be finite and > 0.")

    if abs(x_ohm) < 1e-12:
        return "none", 0.0

    omega = 2.0 * np.pi * freq_hz
    if x_ohm > 0.0:
        return "L", float(x_ohm / omega)
    return "C", float(-1.0 / (omega * x_ohm))


def run_case(case: SmithMatchCase) -> Dict[str, object]:
    """Execute one Smith-chart matching case and return structured results."""
    z_load_norm = normalize_impedance(case.z_load_ohm, case.z0_ohm)
    gamma_load = gamma_from_z_norm(z_load_norm)
    z_roundtrip = z_norm_from_gamma(gamma_load)
    roundtrip_error = abs(z_roundtrip - z_load_norm)

    res_r, res_x = smith_circle_residuals(z_load_norm)

    thetas = find_r_one_intersections(z_load_norm=z_load_norm, grid_points=case.grid_points)
    if len(thetas) == 0:
        raise RuntimeError("No Re(z_in)=1 intersection found in [0, pi].")

    solutions: List[Dict[str, float | str | complex]] = []
    for theta in thetas:
        z_in = input_impedance_norm(z_load_norm, theta)
        x_series_norm = series_reactance_for_match(z_in)
        z_matched = z_in + 1j * x_series_norm

        gamma_rot_formula = gamma_after_line(gamma_load, theta)
        gamma_rot_from_z = gamma_from_z_norm(z_in)
        rotation_error = abs(gamma_rot_formula - gamma_rot_from_z)

        x_series_ohm = x_series_norm * case.z0_ohm
        comp_type, comp_value = reactance_to_component(x_series_ohm, case.freq_hz)

        solutions.append(
            {
                "theta_rad": float(theta),
                "line_len_lambda": float(theta / (2.0 * np.pi)),
                "z_in_norm": z_in,
                "x_series_norm": float(x_series_norm),
                "x_series_ohm": float(x_series_ohm),
                "z_matched_norm": z_matched,
                "match_error": float(abs(z_matched - (1.0 + 0.0j))),
                "rotation_error": float(rotation_error),
                "component_type": comp_type,
                "component_value": float(comp_value),
            }
        )

    return {
        "name": case.name,
        "z0_ohm": float(case.z0_ohm),
        "z_load_ohm": case.z_load_ohm,
        "z_load_norm": z_load_norm,
        "gamma_load": gamma_load,
        "gamma_mag": float(abs(gamma_load)),
        "vswr": float(vswr_from_gamma(gamma_load)),
        "roundtrip_error": float(roundtrip_error),
        "circle_residual_r": float(res_r),
        "circle_residual_x": float(res_x),
        "solutions": solutions,
    }


def _fmt_complex(z: complex) -> str:
    return f"{z.real:.6f}{z.imag:+.6f}j"


def print_case_report(report: Dict[str, object]) -> None:
    """Pretty-print one case result."""
    print(f"\n=== Case: {report['name']} ===")
    print(f"Z0                 : {float(report['z0_ohm']):.3f} ohm")
    print(f"ZL                 : {_fmt_complex(report['z_load_ohm'])} ohm")
    print(f"zL = ZL/Z0         : {_fmt_complex(report['z_load_norm'])}")
    print(f"Gamma_L            : {_fmt_complex(report['gamma_load'])}")
    print(f"|Gamma_L|          : {float(report['gamma_mag']):.6f}")
    print(f"VSWR               : {float(report['vswr']):.6f}")
    print(f"z<->Gamma roundtrip: {float(report['roundtrip_error']):.3e}")
    print(f"r-circle residual  : {float(report['circle_residual_r']):.3e}")
    print(f"x-circle residual  : {float(report['circle_residual_x']):.3e}")

    print("Solutions (line + series reactance):")
    solutions = report["solutions"]
    for idx, sol in enumerate(solutions, start=1):
        comp_type = str(sol["component_type"])
        comp_value = float(sol["component_value"])
        if comp_type == "L":
            comp_text = f"L = {comp_value*1e9:.3f} nH"
        elif comp_type == "C":
            comp_text = f"C = {comp_value*1e12:.3f} pF"
        else:
            comp_text = "none"

        print(f"  [{idx}] theta = {float(sol['theta_rad']):.6f} rad")
        print(f"      line length         : {float(sol['line_len_lambda']):.6f} lambda")
        print(f"      z_in_norm           : {_fmt_complex(sol['z_in_norm'])}")
        print(f"      series x (norm)     : {float(sol['x_series_norm']):.6f}")
        print(f"      series X (ohm)      : {float(sol['x_series_ohm']):.6f}")
        print(f"      component           : {comp_text}")
        print(f"      |z_match-(1+j0)|    : {float(sol['match_error']):.3e}")
        print(f"      gamma rotation err  : {float(sol['rotation_error']):.3e}")


def main() -> None:
    cases = [
        SmithMatchCase(
            name="Case-A (moderately capacitive load)",
            z0_ohm=50.0,
            z_load_ohm=30.0 - 40.0j,
            freq_hz=1.0e9,
            grid_points=6001,
        ),
        SmithMatchCase(
            name="Case-B (inductive load)",
            z0_ohm=50.0,
            z_load_ohm=120.0 + 80.0j,
            freq_hz=2.4e9,
            grid_points=6001,
        ),
    ]

    reports = [run_case(case) for case in cases]
    for report in reports:
        print_case_report(report)

    max_roundtrip_error = max(float(r["roundtrip_error"]) for r in reports)
    max_circle_residual = max(
        max(abs(float(r["circle_residual_r"])), abs(float(r["circle_residual_x"]))) for r in reports
    )

    all_solutions = [sol for r in reports for sol in r["solutions"]]
    max_match_error = max(float(sol["match_error"]) for sol in all_solutions)
    max_rotation_error = max(float(sol["rotation_error"]) for sol in all_solutions)

    pass_flag = (
        max_roundtrip_error < 1e-12
        and max_circle_residual < 1e-10
        and max_match_error < 1e-9
        and max_rotation_error < 1e-10
    )

    print("\n=== Summary ===")
    print(f"max z<->Gamma roundtrip error : {max_roundtrip_error:.3e}")
    print(f"max circle-equation residual  : {max_circle_residual:.3e}")
    print(f"max matching error            : {max_match_error:.3e}")
    print(f"max gamma-rotation error      : {max_rotation_error:.3e}")
    print(f"all checks pass               : {pass_flag}")

    assert pass_flag, "Smith-chart MVP quality gate failed."


if __name__ == "__main__":
    main()
