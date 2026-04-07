"""Minimal runnable MVP for Third Law of Thermodynamics.

This script demonstrates three core Third-Law ideas with a transparent low-temperature model:
1) Perfect crystal entropy approaches zero as T -> 0.
2) Entropy difference between perfect-crystal states vanishes as T -> 0.
3) Absolute zero is unattainable in finite cooling time under an exponential cooling law.

Model used (per mole, synthetic but physically motivated near low temperature):
    C(T) = a*T^3 + b*T
    S(T) = S_res + integral_0^T C(T')/T' dT'

For this model:
    S(T) = S_res + (a/3)*T^3 + b*T
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MaterialModel:
    """Low-temperature heat-capacity model for one material."""

    name: str
    a_cubic: float
    b_linear: float
    s_residual: float

    def validate(self) -> None:
        if self.a_cubic < 0.0:
            raise ValueError(f"a_cubic must be >= 0, got {self.a_cubic}")
        if self.b_linear < 0.0:
            raise ValueError(f"b_linear must be >= 0, got {self.b_linear}")
        if self.s_residual < 0.0:
            raise ValueError(f"s_residual must be >= 0, got {self.s_residual}")

    def heat_capacity(self, t_k: np.ndarray | float) -> np.ndarray:
        """Return C(T) in J/(mol*K)."""
        t = np.asarray(t_k, dtype=np.float64)
        if np.any(t < 0.0):
            raise ValueError("Temperature must be non-negative")
        return self.a_cubic * t**3 + self.b_linear * t

    def entropy_analytic(self, t_k: np.ndarray | float) -> np.ndarray:
        """Closed-form entropy from the model.

        S(T) = S_res + (a/3)*T^3 + b*T
        """
        t = np.asarray(t_k, dtype=np.float64)
        if np.any(t < 0.0):
            raise ValueError("Temperature must be non-negative")
        return self.s_residual + (self.a_cubic / 3.0) * t**3 + self.b_linear * t

    def entropy_numerical(self, t_k: float, n_grid: int = 4000) -> float:
        """Numerically integrate S(T) = S_res + integral_0^T C/T dT.

        The lower bound is shifted to a tiny epsilon to avoid explicit 0/0 evaluation.
        """
        if t_k < 0.0:
            raise ValueError("Temperature must be non-negative")
        if n_grid < 10:
            raise ValueError("n_grid must be >= 10")
        if t_k == 0.0:
            return float(self.s_residual)

        eps = 1e-8
        t_grid = np.linspace(eps, t_k, n_grid, dtype=np.float64)
        integrand = self.heat_capacity(t_grid) / t_grid
        integral = float(np.trapezoid(integrand, t_grid))
        return float(self.s_residual + integral)


def build_entropy_frame(materials: list[MaterialModel], t_grid: np.ndarray) -> pd.DataFrame:
    """Build entropy table for each material and temperature."""
    rows: list[dict[str, float | str]] = []
    for mat in materials:
        mat.validate()
        for t_k in t_grid:
            s_analytic = float(mat.entropy_analytic(t_k))
            s_numerical = float(mat.entropy_numerical(float(t_k), n_grid=5000))
            rows.append(
                {
                    "material": mat.name,
                    "T(K)": float(t_k),
                    "C(T) [J/(mol*K)]": float(mat.heat_capacity(t_k)),
                    "S_analytic [J/(mol*K)]": s_analytic,
                    "S_numerical [J/(mol*K)]": s_numerical,
                    "abs_error": abs(s_analytic - s_numerical),
                }
            )
    return pd.DataFrame(rows)


def build_entropy_gap_frame(
    mat_a: MaterialModel,
    mat_b: MaterialModel,
    t_grid_desc: np.ndarray,
) -> pd.DataFrame:
    """Build |S_A - S_B| table for two perfect-crystal candidates."""
    rows: list[dict[str, float]] = []
    for t_k in t_grid_desc:
        sa = float(mat_a.entropy_analytic(t_k))
        sb = float(mat_b.entropy_analytic(t_k))
        rows.append(
            {
                "T(K)": float(t_k),
                "abs_entropy_gap": abs(sa - sb),
            }
        )
    return pd.DataFrame(rows)


def simulate_exponential_cooling(
    material: MaterialModel,
    t0_k: float,
    tau_s: float,
    dt_s: float,
    n_steps: int,
) -> pd.DataFrame:
    """Finite-time cooling trajectory: T(t) = T0 * exp(-t/tau)."""
    if t0_k <= 0.0:
        raise ValueError("t0_k must be positive")
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    rows: list[dict[str, float]] = []
    for step in range(n_steps + 1):
        t_s = dt_s * step
        t_k = t0_k * np.exp(-t_s / tau_s)
        s_k = float(material.entropy_analytic(float(t_k)))
        rows.append(
            {
                "step": float(step),
                "time(s)": float(t_s),
                "T(K)": float(t_k),
                "S(T) [J/(mol*K)]": s_k,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # Two perfect-crystal variants: residual entropy 0 and no linear term.
    crystal_a = MaterialModel(
        name="PerfectCrystal-A",
        a_cubic=2.0e-3,
        b_linear=0.0,
        s_residual=0.0,
    )
    crystal_b = MaterialModel(
        name="PerfectCrystal-B",
        a_cubic=2.6e-3,
        b_linear=0.0,
        s_residual=0.0,
    )

    # Disordered case: nonzero residual entropy + linear low-T contribution.
    glass_like = MaterialModel(
        name="Disordered-GlassLike",
        a_cubic=1.7e-3,
        b_linear=1.5e-2,
        s_residual=0.35,
    )

    materials = [crystal_a, crystal_b, glass_like]

    t_grid = np.array([0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=np.float64)
    entropy_frame = build_entropy_frame(materials=materials, t_grid=t_grid)

    gap_t_grid = np.array([5.0, 2.0, 1.0, 0.5, 0.2, 0.1], dtype=np.float64)
    gap_frame = build_entropy_gap_frame(crystal_a, crystal_b, gap_t_grid)

    cooling_frame = simulate_exponential_cooling(
        material=crystal_a,
        t0_k=10.0,
        tau_s=30.0,
        dt_s=1.0,
        n_steps=180,
    )
    cooling_view = cooling_frame.iloc[[0, 1, 2, 5, 10, 20, 40, 80, 120, 180]].copy()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Third Law of Thermodynamics MVP")
    print("Model: C(T)=a*T^3+b*T,  S(T)=S_res+integral_0^T C/T dT")

    print("\nEntropy table:")
    print(entropy_frame.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    print("\nEntropy-gap trend between perfect crystals:")
    print(gap_frame.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    print("\nFinite-time cooling trajectory (sampled points):")
    print(cooling_view.to_string(index=False, float_format=lambda x: f"{x:.10f}"))

    # 1) Numerical integration should match analytic entropy very closely.
    max_abs_error = float(entropy_frame["abs_error"].max())
    assert max_abs_error < 2e-6, f"Entropy integration error too large: {max_abs_error}"

    # 2) Third-law (Planck statement for perfect crystal): S(T) -> 0 as T -> 0.
    t_small = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    s_small = crystal_a.entropy_analytic(t_small)
    assert np.all(np.diff(s_small) < 0.0), "S(T) should decrease as T decreases in this probe order"
    assert float(s_small[-1]) < 1e-9, f"Low-T entropy not close to zero: {s_small[-1]}"

    # 3) Nernst heat theorem form: entropy difference between perfect crystals -> 0 as T -> 0.
    gaps = gap_frame["abs_entropy_gap"].to_numpy(dtype=np.float64)
    assert np.all(np.diff(gaps) < 0.0), "Entropy gap should shrink as temperature decreases"
    assert float(gaps[-1]) < 1e-6, f"Entropy gap near 0 K is too large: {gaps[-1]}"

    # 4) Unattainability illustration: finite-time exponential cooling never reaches exactly 0 K.
    temps = cooling_frame["T(K)"].to_numpy(dtype=np.float64)
    assert np.all(temps > 0.0), "Finite-time exponential cooling should remain above 0 K"
    assert np.all(np.diff(temps) < 0.0), "Cooling trajectory should be strictly decreasing"
    assert float(temps[-1]) > 0.0

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
