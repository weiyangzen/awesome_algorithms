"""Minimal runnable MVP for GW approximation (PHYS-0215).

This demo provides an explicit, auditable G0W0 workflow on a two-level model:
1) build independent-particle polarization P0(omega)
2) compute screened interaction W(omega) = v / (1 - v P0)
3) evaluate correlation self-energy Sigma_c(omega) = i/(2pi) * integral G0(omega+omega') * [W-v] d omega'
4) combine with static exchange Sigma_x and solve quasiparticle correction with Z factor
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import simpson


@dataclass(frozen=True)
class StateSpec:
    """Single-particle state used by the toy G0W0 solver."""

    name: str
    eps_ks: float
    v_xc: float
    occupied: bool
    sigma_x: float


@dataclass(frozen=True)
class GWConfig:
    """Configuration for a minimal two-level G0W0 model."""

    eps_valence: float = -5.4
    eps_conduction: float = -1.9
    vxc_valence: float = -0.62
    vxc_conduction: float = -1.05

    bare_coulomb: float = 1.15
    transition_strength: float = 0.52

    fock_u_vv: float = 0.95
    fock_u_cv: float = 0.58

    omega_max: float = 30.0
    n_omega: int = 6001
    eta: float = 0.18
    derivative_step: float = 0.03

    @property
    def states(self) -> tuple[StateSpec, StateSpec]:
        return (
            StateSpec(
                name="valence",
                eps_ks=self.eps_valence,
                v_xc=self.vxc_valence,
                occupied=True,
                sigma_x=-self.fock_u_vv,
            ),
            StateSpec(
                name="conduction",
                eps_ks=self.eps_conduction,
                v_xc=self.vxc_conduction,
                occupied=False,
                sigma_x=-self.fock_u_cv,
            ),
        )


def build_frequency_grid(cfg: GWConfig) -> np.ndarray:
    if cfg.omega_max <= 0.0:
        raise ValueError("omega_max must be positive")
    if cfg.n_omega < 101:
        raise ValueError("n_omega must be >= 101")
    if cfg.n_omega % 2 == 0:
        raise ValueError("n_omega must be odd for Simpson integration")
    return np.linspace(-cfg.omega_max, cfg.omega_max, cfg.n_omega, dtype=float)


def independent_particle_polarization(
    omega: np.ndarray,
    transition_energy: float,
    transition_strength: float,
    eta: float,
) -> np.ndarray:
    """Independent-particle P0 for one dominant v->c transition."""
    if transition_energy <= 0.0:
        raise ValueError("transition_energy must be positive")
    if transition_strength <= 0.0:
        raise ValueError("transition_strength must be positive")

    pref = 2.0 * transition_strength**2
    term_resonant = 1.0 / (omega - transition_energy + 1j * eta)
    term_antires = 1.0 / (omega + transition_energy - 1j * eta)
    return pref * (term_resonant - term_antires)


def screened_interaction(bare_coulomb: float, polarization: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RPA screened interaction W(omega) and denominator D = 1 - vP0."""
    if bare_coulomb <= 0.0:
        raise ValueError("bare_coulomb must be positive")

    denom = 1.0 - bare_coulomb * polarization
    if float(np.min(np.abs(denom))) < 1.0e-10:
        raise ValueError("Screening denominator is too close to zero; unstable parameter set")

    w = bare_coulomb / denom
    return w, denom


def correlation_self_energy(
    omega_eval: float,
    state: StateSpec,
    omega_grid: np.ndarray,
    w_corr: np.ndarray,
    eta: float,
) -> complex:
    """Numerical real-axis integral for Sigma_c(omega_eval)."""
    sign = -1.0 if state.occupied else 1.0
    g0 = 1.0 / (omega_eval + omega_grid - state.eps_ks + 1j * sign * eta)
    integrand = g0 * w_corr
    return 1j * simpson(integrand, x=omega_grid) / (2.0 * np.pi)


def total_self_energy(
    omega_eval: float,
    state: StateSpec,
    omega_grid: np.ndarray,
    w_corr: np.ndarray,
    eta: float,
) -> complex:
    return state.sigma_x + correlation_self_energy(
        omega_eval=omega_eval,
        state=state,
        omega_grid=omega_grid,
        w_corr=w_corr,
        eta=eta,
    )


def quasiparticle_update(
    state: StateSpec,
    omega_grid: np.ndarray,
    w_corr: np.ndarray,
    cfg: GWConfig,
) -> dict[str, float | str]:
    """One-shot G0W0 correction around KS energy with linearized Z factor."""
    h = cfg.derivative_step

    sigma_0 = total_self_energy(state.eps_ks, state, omega_grid, w_corr, cfg.eta)
    sigma_p = total_self_energy(state.eps_ks + h, state, omega_grid, w_corr, cfg.eta)
    sigma_m = total_self_energy(state.eps_ks - h, state, omega_grid, w_corr, cfg.eta)

    d_re_sigma = (sigma_p.real - sigma_m.real) / (2.0 * h)
    z_factor = 1.0 / (1.0 - d_re_sigma)

    e_qp = state.eps_ks + z_factor * (sigma_0.real - state.v_xc)
    sigma_c = sigma_0 - state.sigma_x

    return {
        "state": state.name,
        "occupied": float(1.0 if state.occupied else 0.0),
        "eps_KS": float(state.eps_ks),
        "v_xc": float(state.v_xc),
        "Sigma_x": float(state.sigma_x),
        "ReSigma_c": float(sigma_c.real),
        "ReSigma_total": float(sigma_0.real),
        "ImSigma_total": float(sigma_0.imag),
        "dReSigma_domega": float(d_re_sigma),
        "Z": float(z_factor),
        "E_QP": float(e_qp),
        "QP_minus_KS": float(e_qp - state.eps_ks),
    }


def run_g0w0(cfg: GWConfig) -> dict[str, object]:
    omega_grid = build_frequency_grid(cfg)

    transition_energy = cfg.eps_conduction - cfg.eps_valence
    p0 = independent_particle_polarization(
        omega=omega_grid,
        transition_energy=transition_energy,
        transition_strength=cfg.transition_strength,
        eta=cfg.eta,
    )
    w, denom = screened_interaction(cfg.bare_coulomb, p0)
    w_corr = w - cfg.bare_coulomb

    rows = [
        quasiparticle_update(state=st, omega_grid=omega_grid, w_corr=w_corr, cfg=cfg)
        for st in cfg.states
    ]
    table = pd.DataFrame(rows)

    idx0 = len(omega_grid) // 2
    w0 = w[idx0]
    screening_ratio = float(w0.real / cfg.bare_coulomb)

    ks_gap = float(cfg.eps_conduction - cfg.eps_valence)
    e_qp_v = float(table.loc[table["state"] == "valence", "E_QP"].iloc[0])
    e_qp_c = float(table.loc[table["state"] == "conduction", "E_QP"].iloc[0])
    qp_gap = float(e_qp_c - e_qp_v)

    return {
        "table": table,
        "omega_grid": omega_grid,
        "p0": p0,
        "w": w,
        "denom": denom,
        "screening_ratio": screening_ratio,
        "w0_real": float(w0.real),
        "w0_imag": float(w0.imag),
        "ks_gap": ks_gap,
        "qp_gap": qp_gap,
        "cfg": cfg,
    }


def main() -> None:
    cfg = GWConfig()
    result = run_g0w0(cfg)
    table = result["table"]

    valence_row = table[table["state"] == "valence"].iloc[0]
    conduction_row = table[table["state"] == "conduction"].iloc[0]

    min_abs_denom = float(np.min(np.abs(result["denom"])))

    checks = {
        "screening denominator min|1-vP0| > 5e-2": min_abs_denom > 5.0e-2,
        "static screening ratio in (0,1)": 0.0 < result["screening_ratio"] < 1.0,
        "all Z factors in (0, 1.2)": bool(((table["Z"] > 0.0) & (table["Z"] < 1.2)).all()),
        "QP gap > KS gap": result["qp_gap"] > result["ks_gap"],
        "|Im Sigma| < 5e-2 for both states": bool((table["ImSigma_total"].abs() < 5.0e-2).all()),
        "valence shifts downward": valence_row["E_QP"] < valence_row["eps_KS"],
        "conduction shifts upward": conduction_row["E_QP"] > conduction_row["eps_KS"],
    }

    pd.set_option("display.float_format", lambda v: f"{v:.8f}")

    print("=== GW Approximation MVP (G0W0, PHYS-0215) ===")
    print(
        f"levels: eps_v={cfg.eps_valence:.4f} eV, eps_c={cfg.eps_conduction:.4f} eV; "
        f"transition={cfg.eps_conduction - cfg.eps_valence:.4f} eV"
    )
    print(
        f"frequency grid: [-{cfg.omega_max:.1f}, {cfg.omega_max:.1f}] with {cfg.n_omega} points; "
        f"eta={cfg.eta:.3f}, bare_v={cfg.bare_coulomb:.3f}, M={cfg.transition_strength:.3f}"
    )

    print("\nState-wise quasiparticle update:")
    print(table.to_string(index=False))

    screening_summary = pd.DataFrame(
        {
            "quantity": [
                "W(0) real",
                "W(0) imag",
                "W(0)/v",
                "min|1-vP0|",
                "KS gap",
                "QP gap",
                "gap opening (QP-KS)",
            ],
            "value": [
                result["w0_real"],
                result["w0_imag"],
                result["screening_ratio"],
                min_abs_denom,
                result["ks_gap"],
                result["qp_gap"],
                result["qp_gap"] - result["ks_gap"],
            ],
        }
    )

    print("\nScreening and gap summary:")
    print(screening_summary.to_string(index=False))

    print("\nThreshold checks:")
    for name, ok in checks.items():
        print(f"- {name}: {'OK' if ok else 'FAIL'}")

    if all(checks.values()):
        print("\nValidation: PASS")
    else:
        print("\nValidation: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
