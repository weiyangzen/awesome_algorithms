"""Minimal runnable MVP for Second Harmonic Generation (SHG).

The model uses coupled-wave equations with complex field amplitudes and solves
an initial value problem via SciPy's adaptive RK45 integrator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SHGParams:
    """Configuration for 1D SHG propagation in a lossless/lossy medium."""

    kappa: float = 1.0
    delta_k: float = 0.0
    alpha1: float = 0.0
    alpha2: float = 0.0
    length: float = 10.0
    n_points: int = 800
    a1_0: complex = 1.0 + 0.0j
    a2_0: complex = 0.0 + 0.0j


def _pack_complex(a1: complex, a2: complex) -> np.ndarray:
    return np.array([a1.real, a1.imag, a2.real, a2.imag], dtype=float)


def _unpack_complex(y: np.ndarray) -> tuple[complex, complex]:
    a1 = y[0] + 1j * y[1]
    a2 = y[2] + 1j * y[3]
    return a1, a2


def shg_rhs(z: float, y: np.ndarray, params: SHGParams) -> np.ndarray:
    """Right-hand side of the coupled SHG ODE system."""
    a1, a2 = _unpack_complex(y)
    phase = np.exp(1j * params.delta_k * z)

    d_a1 = -1j * params.kappa * np.conj(a1) * a2 * np.conj(phase)
    d_a2 = -1j * params.kappa * (a1**2) * phase

    if params.alpha1:
        d_a1 -= 0.5 * params.alpha1 * a1
    if params.alpha2:
        d_a2 -= 0.5 * params.alpha2 * a2

    return _pack_complex(d_a1, d_a2)


def simulate_shg(params: SHGParams) -> dict[str, np.ndarray]:
    """Run one SHG simulation and return trajectories and diagnostics."""
    z_grid = np.linspace(0.0, params.length, params.n_points)
    y0 = _pack_complex(params.a1_0, params.a2_0)

    sol = solve_ivp(
        fun=lambda z, y: shg_rhs(z, y, params),
        t_span=(0.0, params.length),
        y0=y0,
        method="RK45",
        t_eval=z_grid,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    a1 = sol.y[0] + 1j * sol.y[1]
    a2 = sol.y[2] + 1j * sol.y[3]

    i1 = np.abs(a1) ** 2
    i2 = np.abs(a2) ** 2
    total = i1 + i2
    conversion = i2 / np.maximum(total, 1e-12)

    return {
        "z": sol.t,
        "a1": a1,
        "a2": a2,
        "i1": i1,
        "i2": i2,
        "total": total,
        "conversion": conversion,
    }


def summarize_case(delta_k: float, result: dict[str, np.ndarray]) -> dict[str, float]:
    """Extract compact metrics from one run."""
    z = result["z"]
    i2 = result["i2"]
    total = result["total"]
    conversion = result["conversion"]

    peak_idx = int(np.argmax(i2))
    drift = np.max(np.abs(total - total[0]))

    return {
        "delta_k": float(delta_k),
        "peak_i2": float(i2[peak_idx]),
        "z_at_peak": float(z[peak_idx]),
        "final_i2": float(i2[-1]),
        "final_conversion": float(conversion[-1]),
        "power_drift": float(drift),
    }


def build_trace_table(result: dict[str, np.ndarray], n_rows: int = 8) -> pd.DataFrame:
    """Build a small sampled table for human inspection."""
    z = result["z"]
    indices = np.linspace(0, len(z) - 1, n_rows, dtype=int)
    return pd.DataFrame(
        {
            "z": result["z"][indices],
            "I_w": result["i1"][indices],
            "I_2w": result["i2"][indices],
            "eta": result["conversion"][indices],
        }
    )


def main() -> None:
    base = SHGParams(kappa=1.0, alpha1=0.0, alpha2=0.0, length=10.0, n_points=1000)
    mismatch_values = [0.0, 1.5, 4.0, 8.0]

    records: list[dict[str, float]] = []
    traces: dict[float, dict[str, np.ndarray]] = {}
    for delta_k in mismatch_values:
        params = replace(base, delta_k=delta_k)
        result = simulate_shg(params)
        traces[delta_k] = result
        records.append(summarize_case(delta_k, result))

    summary_df = pd.DataFrame(records).sort_values("delta_k").reset_index(drop=True)
    phase_matched_trace = build_trace_table(traces[0.0], n_rows=8)

    output_dir = Path(__file__).resolve().parent
    summary_path = output_dir / "results_summary.csv"
    trace_path = output_dir / "phase_matched_trace.csv"
    summary_df.to_csv(summary_path, index=False)
    phase_matched_trace.to_csv(trace_path, index=False)

    print("=== SHG sweep summary (smaller delta_k -> easier phase matching) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print("\n=== Sampled phase-matched trajectory (delta_k=0) ===")
    print(phase_matched_trace.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    print(f"\nSaved: {summary_path.name}, {trace_path.name}")


if __name__ == "__main__":
    main()
