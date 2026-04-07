"""Q-switching laser rate-equation MVP.

This script simulates a single actively Q-switched pulse with a
piecewise cavity loss model:
- Before switch: high cavity loss (low Q), photons cannot build up.
- After switch: low cavity loss (high Q), stored inversion is released
  as a short high-peak pulse.

The model uses two coupled rate equations:
  dN/dt   = R_p - N/tau_f - g * N * Phi
  dPhi/dt = (g * N - 1/tau_p(t)) * Phi + beta_sp * N/tau_f

where N is inversion (normalized), Phi is intracavity photon density
(arbitrary normalized unit), and tau_p(t) is the time-varying cavity
photon lifetime controlled by the Q-switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class QSwitchParams:
    """Physical/numerical parameters for the minimal Q-switching model."""

    pump_rate: float = 5.0e4
    tau_f: float = 230e-6
    gain_coeff: float = 2.0e7
    tau_p_closed: float = 1.0e-9
    tau_p_open: float = 10.0e-9
    beta_sp: float = 1.0e-7
    t_switch: float = 180e-6
    t_end: float = 240e-6
    dt: float = 0.5e-9
    n0: float = 0.0
    phi0: float = 1.0e-12


def cavity_photon_lifetime(t: np.ndarray, params: QSwitchParams) -> np.ndarray:
    """Piecewise photon lifetime profile set by the Q-switch timing."""

    return np.where(t < params.t_switch, params.tau_p_closed, params.tau_p_open)


def simulate_q_switch(params: QSwitchParams) -> Dict[str, np.ndarray]:
    """Integrate the coupled rate equations with explicit Euler."""

    n_steps = int(np.floor(params.t_end / params.dt)) + 1
    t = np.linspace(0.0, params.t_end, n_steps, dtype=float)
    tau_p = cavity_photon_lifetime(t, params)

    n = np.zeros(n_steps, dtype=float)
    phi = np.zeros(n_steps, dtype=float)
    n[0] = params.n0
    phi[0] = params.phi0

    for i in range(n_steps - 1):
        dn_dt = (
            params.pump_rate
            - n[i] / params.tau_f
            - params.gain_coeff * n[i] * phi[i]
        )
        dphi_dt = (
            (params.gain_coeff * n[i] - 1.0 / tau_p[i]) * phi[i]
            + params.beta_sp * n[i] / params.tau_f
        )

        n_next = n[i] + params.dt * dn_dt
        phi_next = phi[i] + params.dt * dphi_dt

        # Numerical guard: this MVP keeps state variables non-negative.
        n[i + 1] = max(n_next, 0.0)
        phi[i + 1] = max(phi_next, 0.0)

    return {"t": t, "tau_p": tau_p, "n": n, "phi": phi}


def estimate_fwhm(t: np.ndarray, y: np.ndarray) -> float:
    """Return FWHM in seconds; 0.0 if the signal has no resolvable half-max span."""

    peak = float(np.max(y))
    if peak <= 0.0:
        return 0.0

    half = 0.5 * peak
    idx = np.where(y >= half)[0]
    if idx.size < 2:
        return 0.0

    return float(t[idx[-1]] - t[idx[0]])


def extract_pulse_metrics(results: Dict[str, np.ndarray], params: QSwitchParams) -> Dict[str, float]:
    """Compute key pulse metrics after the Q-switch opens."""

    t = results["t"]
    n = results["n"]
    phi = results["phi"]

    post_mask = t >= params.t_switch
    pre_mask = ~post_mask
    t_post = t[post_mask]
    phi_post = phi[post_mask]

    peak_idx_local = int(np.argmax(phi_post))
    peak_phi = float(phi_post[peak_idx_local])
    peak_time = float(t_post[peak_idx_local])

    fwhm = estimate_fwhm(t_post, phi_post)
    pulse_area = float(np.trapezoid(phi_post, t_post))

    switch_index = int(np.searchsorted(t, params.t_switch, side="left"))
    switch_index = max(1, min(switch_index, t.size - 1))

    return {
        "inversion_before_switch": float(n[switch_index - 1]),
        "max_photon_before_switch": float(np.max(phi[pre_mask])) if np.any(pre_mask) else 0.0,
        "peak_photon": peak_phi,
        "peak_time_s": peak_time,
        "peak_delay_s": float(peak_time - params.t_switch),
        "fwhm_s": fwhm,
        "pulse_area": pulse_area,
        "final_inversion": float(n[-1]),
        "final_photon": float(phi[-1]),
    }


def format_metrics(metrics: Dict[str, float]) -> Tuple[str, ...]:
    """Create a compact printable summary."""

    return (
        f"Inversion just before switch: {metrics['inversion_before_switch']:.6f}",
        f"Max photon before switch:     {metrics['max_photon_before_switch']:.6e}",
        f"Peak photon after switch:     {metrics['peak_photon']:.6e}",
        f"Peak delay after switch:      {metrics['peak_delay_s'] * 1e9:.2f} ns",
        f"Pulse FWHM:                   {metrics['fwhm_s'] * 1e9:.2f} ns",
        f"Pulse area (post-switch):     {metrics['pulse_area']:.6e}",
        f"Final inversion:              {metrics['final_inversion']:.6f}",
        f"Final photon:                 {metrics['final_photon']:.6e}",
    )


def run_demo() -> None:
    params = QSwitchParams()
    results = simulate_q_switch(params)
    metrics = extract_pulse_metrics(results, params)

    print("=== Q-Switching Laser MVP (Rate Equations) ===")
    print(f"Grid points: {results['t'].size}")
    print(f"dt: {params.dt * 1e9:.3f} ns")
    print(f"Switch time: {params.t_switch * 1e6:.3f} us")
    print()
    for line in format_metrics(metrics):
        print(line)

    # Basic sanity checks for this configured demonstration case.
    assert metrics["max_photon_before_switch"] < 1e-8, "Photon leakage too high before opening Q."
    assert metrics["peak_photon"] > 1e-3, "Pulse peak is unexpectedly weak."
    assert metrics["peak_delay_s"] >= 0.0, "Peak should occur after switch opening."

    print("\nAll checks passed.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
