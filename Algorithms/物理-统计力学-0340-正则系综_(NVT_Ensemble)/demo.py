"""NVT ensemble MVP via Metropolis sampling on a 1D harmonic oscillator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NVTConfig:
    mass: float = 1.5
    spring_k: float = 2.0
    temperatures: tuple[float, ...] = (0.8, 1.2, 2.0)
    n_walkers: int = 384
    burn_in: int = 1600
    sample_steps: int = 950
    thin: int = 5
    proposal_scale: float = 1.10
    seed: int = 20260407


def hamiltonian(q: np.ndarray, p: np.ndarray, mass: float, spring_k: float) -> np.ndarray:
    """Compute H(q,p)=p^2/(2m)+kq^2/2 for vectorized states."""
    return 0.5 * (p * p) / mass + 0.5 * spring_k * (q * q)


def theoretical_observables(temperature: float, mass: float, spring_k: float) -> dict[str, float]:
    """Analytical canonical expectations for the classical 1D harmonic oscillator."""
    return {
        "q2_theory": temperature / spring_k,
        "p2_theory": mass * temperature,
        "energy_theory": temperature,
        "cv_theory": 1.0,
    }


def run_metropolis_nvt(
    temperature: float,
    cfg: NVTConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run one temperature point of canonical (NVT) sampling."""
    beta = 1.0 / temperature
    n_walkers = cfg.n_walkers
    total_steps = cfg.burn_in + cfg.sample_steps * cfg.thin

    q_sigma_eq = np.sqrt(temperature / cfg.spring_k)
    p_sigma_eq = np.sqrt(cfg.mass * temperature)
    q = rng.normal(0.0, q_sigma_eq, size=n_walkers)
    p = rng.normal(0.0, p_sigma_eq, size=n_walkers)

    proposal_q = cfg.proposal_scale * q_sigma_eq
    proposal_p = cfg.proposal_scale * p_sigma_eq

    accepted = 0
    attempted = 0

    sum_q2 = 0.0
    sum_p2 = 0.0
    energy_samples = np.empty((cfg.sample_steps, n_walkers), dtype=np.float64)
    saved = 0

    for step in range(total_steps):
        q_prop = q + rng.normal(0.0, proposal_q, size=n_walkers)
        p_prop = p + rng.normal(0.0, proposal_p, size=n_walkers)

        current_energy = hamiltonian(q, p, cfg.mass, cfg.spring_k)
        proposed_energy = hamiltonian(q_prop, p_prop, cfg.mass, cfg.spring_k)
        delta_e = proposed_energy - current_energy

        uniform = rng.random(n_walkers)
        accept = (delta_e <= 0.0) | (uniform < np.exp(-beta * delta_e))

        q = np.where(accept, q_prop, q)
        p = np.where(accept, p_prop, p)
        accepted += int(np.count_nonzero(accept))
        attempted += n_walkers

        if step >= cfg.burn_in and (step - cfg.burn_in) % cfg.thin == 0:
            snapshot_energy = hamiltonian(q, p, cfg.mass, cfg.spring_k)
            energy_samples[saved] = snapshot_energy
            sum_q2 += float(np.mean(q * q))
            sum_p2 += float(np.mean(p * p))
            saved += 1

    if saved != cfg.sample_steps:
        raise RuntimeError(f"Saved sample count mismatch: {saved} != {cfg.sample_steps}")

    q2_emp = sum_q2 / cfg.sample_steps
    p2_emp = sum_p2 / cfg.sample_steps
    energy_emp = float(np.mean(energy_samples))
    cv_emp = float(np.var(energy_samples, ddof=1) / (temperature * temperature))
    acceptance_rate = accepted / attempted

    theory = theoretical_observables(temperature, cfg.mass, cfg.spring_k)

    rel_errors = [
        abs((q2_emp - theory["q2_theory"]) / theory["q2_theory"]),
        abs((p2_emp - theory["p2_theory"]) / theory["p2_theory"]),
        abs((energy_emp - theory["energy_theory"]) / theory["energy_theory"]),
        abs((cv_emp - theory["cv_theory"]) / theory["cv_theory"]),
    ]

    return {
        "temperature": temperature,
        "acceptance_rate": acceptance_rate,
        "q2_emp": q2_emp,
        "q2_theory": theory["q2_theory"],
        "p2_emp": p2_emp,
        "p2_theory": theory["p2_theory"],
        "energy_emp": energy_emp,
        "energy_theory": theory["energy_theory"],
        "cv_emp": cv_emp,
        "cv_theory": theory["cv_theory"],
        "max_relative_error": max(rel_errors),
    }


def main() -> None:
    cfg = NVTConfig()
    rows: list[dict[str, float]] = []
    for idx, temperature in enumerate(cfg.temperatures):
        run_rng = np.random.default_rng(cfg.seed + 1009 * idx)
        rows.append(run_metropolis_nvt(temperature, cfg, run_rng))

    df = pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("NVT ensemble check on classical 1D harmonic oscillator")
        print(df.round(6).to_string(index=False))

    max_err = float(df["max_relative_error"].max())
    min_acc = float(df["acceptance_rate"].min())
    max_acc = float(df["acceptance_rate"].max())

    print(
        "\nSummary:"
        f" max_relative_error={max_err:.4f},"
        f" acceptance_range=[{min_acc:.3f}, {max_acc:.3f}]"
    )

    # Loose but meaningful tolerances for stochastic validation.
    assert max_err < 0.12, f"Relative error too large: {max_err:.4f}"
    assert 0.20 < min_acc < 0.90 and 0.20 < max_acc < 0.90, (
        "Acceptance rate out of healthy range; tune proposal_scale."
    )


if __name__ == "__main__":
    main()
