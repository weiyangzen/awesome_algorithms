"""Minimal runnable MVP for Gamow-Teller transition calculations.

This script demonstrates a small, transparent pipeline for allowed beta decays:
1) check Gamow-Teller selection rules,
2) compute reduced GT strength B(GT),
3) numerically integrate phase-space factor f(Z, Q),
4) estimate comparative half-life ft and physical half-life t1/2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import integrate

ALPHA = 1.0 / 137.035999084  # fine-structure constant
M_E_MEV = 0.51099895  # electron rest energy (MeV)
K_CONSTANT_S = 6147.0  # beta-decay constant K in seconds
G_A_OVER_G_V = 1.27  # effective axial/vector coupling ratio for this MVP


@dataclass(frozen=True)
class GTTransition:
    """A small record for one candidate beta transition."""

    nuclide: str
    Z_daughter: int
    Q_MeV: float
    J_i: float
    J_f: float
    parity_i: int  # +1 or -1
    parity_f: int  # +1 or -1
    M_GT: float
    mode: str = "beta-"  # beta- or beta+


def _close_to(x: float, target: float, atol: float = 1e-12) -> bool:
    return abs(x - target) <= atol


def is_allowed_gt(transition: GTTransition) -> tuple[bool, str]:
    """Gamow-Teller allowed selection rules: ΔJ = 0,1 (but not 0->0), no parity change."""

    if transition.parity_i not in (-1, 1) or transition.parity_f not in (-1, 1):
        return False, "invalid parity encoding"

    if transition.parity_i != transition.parity_f:
        return False, "parity changes (GT allowed requires parity conserved)"

    delta_j = abs(transition.J_f - transition.J_i)
    if not (_close_to(delta_j, 0.0) or _close_to(delta_j, 1.0)):
        return False, f"|ΔJ|={delta_j:.1f} not in {{0,1}}"

    if _close_to(transition.J_i, 0.0) and _close_to(transition.J_f, 0.0):
        return False, "0->0 excluded for rank-1 GT operator"

    if transition.Q_MeV <= 0.0:
        return False, "Q <= 0, no phase space"

    return True, "allowed"


def reduced_b_gt(M_GT: float, J_i: float) -> float:
    """B(GT) reduced strength without coupling constants.

    B(GT) = |M_GT|^2 / (2J_i + 1)
    """

    denom = 2.0 * J_i + 1.0
    if denom <= 0:
        raise ValueError("2*J_i + 1 must be positive")
    return (M_GT * M_GT) / denom


def fermi_coulomb_correction(Z: int, W: float, mode: str) -> float:
    """Simple Fermi-function approximation F ~ 2*pi*eta / (1 - exp(-2*pi*eta))."""

    p2 = max(W * W - 1.0, 1e-18)
    p = math.sqrt(p2)
    eta = ALPHA * Z * W / p

    if mode == "beta+":
        eta = -eta
    elif mode != "beta-":
        raise ValueError("mode must be 'beta-' or 'beta+'")

    x = 2.0 * math.pi * eta
    if abs(x) < 1e-8:
        return 1.0 + 0.5 * x

    denom = 1.0 - math.exp(-x)
    if abs(denom) < 1e-14:
        return 1.0
    return x / denom


def phase_space_factor_f(Z_daughter: int, Q_MeV: float, mode: str) -> float:
    """Dimensionless allowed-decay phase-space factor.

    f(Z, Q) = integral_1^W0 F(Z, W) * p * W * (W0 - W)^2 dW,
    where W is total lepton energy in electron-mass units.
    """

    W0 = 1.0 + Q_MeV / M_E_MEV
    if W0 <= 1.0:
        return 0.0

    def integrand(W: float) -> float:
        p = math.sqrt(max(W * W - 1.0, 0.0))
        F = fermi_coulomb_correction(Z_daughter, W, mode)
        return F * p * W * (W0 - W) ** 2

    value, _abserr = integrate.quad(
        integrand,
        1.0,
        W0,
        epsabs=1e-10,
        epsrel=1e-8,
        limit=200,
    )
    return max(value, 0.0)


def estimate_half_life(transition: GTTransition) -> dict[str, float | str | bool]:
    """Evaluate one transition under a pure-allowed GT model."""

    allowed, reason = is_allowed_gt(transition)
    out: dict[str, float | str | bool] = {
        "nuclide": transition.nuclide,
        "allowed_gt": allowed,
        "reason": reason,
    }

    if not allowed:
        out.update({
            "B_GT": np.nan,
            "f": np.nan,
            "ft_s": np.nan,
            "t1_2_s": np.nan,
        })
        return out

    B_gt = reduced_b_gt(transition.M_GT, transition.J_i)
    f_val = phase_space_factor_f(transition.Z_daughter, transition.Q_MeV, transition.mode)

    if B_gt <= 0.0 or f_val <= 0.0:
        out.update({
            "B_GT": B_gt,
            "f": f_val,
            "ft_s": np.nan,
            "t1_2_s": np.nan,
        })
        return out

    ft_s = K_CONSTANT_S / (G_A_OVER_G_V**2 * B_gt)
    t12_s = ft_s / f_val

    out.update({
        "B_GT": B_gt,
        "f": f_val,
        "ft_s": ft_s,
        "t1_2_s": t12_s,
    })
    return out


def main() -> None:
    sample = [
        GTTransition(
            nuclide="6He -> 6Li",
            Z_daughter=3,
            Q_MeV=3.508,
            J_i=0.0,
            J_f=1.0,
            parity_i=+1,
            parity_f=+1,
            M_GT=2.0,
            mode="beta-",
        ),
        GTTransition(
            nuclide="14C -> 14N",
            Z_daughter=7,
            Q_MeV=0.156,
            J_i=0.0,
            J_f=1.0,
            parity_i=+1,
            parity_f=+1,
            M_GT=0.002,
            mode="beta-",
        ),
        GTTransition(
            nuclide="100Sn -> 100In",
            Z_daughter=49,
            Q_MeV=3.000,
            J_i=0.0,
            J_f=1.0,
            parity_i=+1,
            parity_f=+1,
            M_GT=1.8,
            mode="beta+",
        ),
        GTTransition(
            nuclide="16N -> 16O (example forbidden)",
            Z_daughter=8,
            Q_MeV=10.420,
            J_i=2.0,
            J_f=0.0,
            parity_i=-1,
            parity_f=+1,
            M_GT=1.0,
            mode="beta-",
        ),
    ]

    rows = [estimate_half_life(t) for t in sample]
    df = pd.DataFrame(rows)

    finite_mask = np.isfinite(df["t1_2_s"].to_numpy(dtype=float))
    log_t = np.full(df.shape[0], np.nan)
    log_t[finite_mask] = np.log10(df.loc[finite_mask, "t1_2_s"].to_numpy(dtype=float))
    df["log10_t1_2_s"] = log_t

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x: .4e}")

    print("Gamow-Teller transition MVP (pure allowed model):")
    print(df[["nuclide", "allowed_gt", "reason", "B_GT", "f", "ft_s", "t1_2_s", "log10_t1_2_s"]])


if __name__ == "__main__":
    main()
