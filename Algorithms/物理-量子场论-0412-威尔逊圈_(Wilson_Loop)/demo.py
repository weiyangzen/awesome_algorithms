"""Minimal runnable MVP for Wilson loop measurement on a 2D U(1) lattice.

This script implements a transparent lattice gauge Monte Carlo pipeline:
1) Wilson plaquette action,
2) local Metropolis updates for link angles,
3) Wilson loop measurements on rectangular contours,
4) simple area-law based string tension estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WilsonLoopConfig:
    lattice_size: int = 8
    beta: float = 1.10
    proposal_width: float = 0.70
    thermalization_sweeps: int = 250
    measurement_sweeps: int = 160
    sweeps_between_measurements: int = 4
    seed: int = 20260407
    loop_shapes: tuple[tuple[int, int], ...] = (
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
    )


def validate_config(config: WilsonLoopConfig) -> None:
    if config.lattice_size < 4:
        raise ValueError("lattice_size must be >= 4.")
    if config.beta <= 0.0:
        raise ValueError("beta must be positive.")
    if not (0.0 < config.proposal_width <= np.pi):
        raise ValueError("proposal_width must be in (0, pi].")
    if config.thermalization_sweeps <= 0:
        raise ValueError("thermalization_sweeps must be positive.")
    if config.measurement_sweeps <= 1:
        raise ValueError("measurement_sweeps must be > 1 for uncertainty estimates.")
    if config.sweeps_between_measurements <= 0:
        raise ValueError("sweeps_between_measurements must be positive.")
    if not config.loop_shapes:
        raise ValueError("loop_shapes cannot be empty.")
    for r, t in config.loop_shapes:
        if r <= 0 or t <= 0:
            raise ValueError("All loop shapes (R, T) must be positive.")
        if r >= config.lattice_size or t >= config.lattice_size:
            raise ValueError("Loop shape must be smaller than lattice size.")


def wrap_angle(theta: float) -> float:
    """Map angle to [-pi, pi)."""
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def plaquette_angle(links: np.ndarray, x: int, y: int) -> float:
    """Return oriented plaquette angle at site (x, y) for 2D U(1)."""
    l = links.shape[0]
    theta = (
        links[x, y, 0]
        + links[(x + 1) % l, y, 1]
        - links[x, (y + 1) % l, 0]
        - links[x, y, 1]
    )
    return float(theta)


def affected_plaquettes(l: int, x: int, y: int, mu: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Plaquettes touching one link in 2D.

    mu = 0: x-direction link at (x, y)
    mu = 1: y-direction link at (x, y)
    """
    if mu == 0:
        return (x, y), (x, (y - 1) % l)
    if mu == 1:
        return (x, y), ((x - 1) % l, y)
    raise ValueError("mu must be 0 or 1.")


def metropolis_sweep(
    links: np.ndarray,
    beta: float,
    proposal_width: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Perform one full-lattice sweep of local Metropolis updates."""
    l = links.shape[0]
    accepted = 0
    total = l * l * 2

    for x in range(l):
        for y in range(l):
            for mu in (0, 1):
                old_theta = float(links[x, y, mu])
                p1, p2 = affected_plaquettes(l=l, x=x, y=y, mu=mu)

                old_cos_sum = np.cos(plaquette_angle(links, *p1)) + np.cos(plaquette_angle(links, *p2))

                proposal = wrap_angle(old_theta + rng.uniform(-proposal_width, proposal_width))
                links[x, y, mu] = proposal

                new_cos_sum = np.cos(plaquette_angle(links, *p1)) + np.cos(plaquette_angle(links, *p2))
                delta_s = -beta * (new_cos_sum - old_cos_sum)

                if delta_s <= 0.0 or rng.random() < np.exp(-delta_s):
                    accepted += 1
                else:
                    links[x, y, mu] = old_theta

    return accepted, total


def average_plaquette(links: np.ndarray) -> float:
    """Compute average plaquette value <cos(theta_p)> on current configuration."""
    theta_p = (
        links[:, :, 0]
        + np.roll(links[:, :, 1], shift=-1, axis=0)
        - np.roll(links[:, :, 0], shift=-1, axis=1)
        - links[:, :, 1]
    )
    return float(np.mean(np.cos(theta_p)))


def loop_phase_at_start(links: np.ndarray, x: int, y: int, r: int, t: int) -> float:
    """Oriented sum of link angles around an R x T rectangle from start (x, y)."""
    l = links.shape[0]
    cx, cy = x, y
    phase = 0.0

    for _ in range(r):
        phase += links[cx, cy, 0]
        cx = (cx + 1) % l

    for _ in range(t):
        phase += links[cx, cy, 1]
        cy = (cy + 1) % l

    for _ in range(r):
        cx = (cx - 1) % l
        phase -= links[cx, cy, 0]

    for _ in range(t):
        cy = (cy - 1) % l
        phase -= links[cx, cy, 1]

    return phase


def measure_wilson_loop(links: np.ndarray, r: int, t: int) -> complex:
    """Average Wilson loop W(R,T) over all starting positions on periodic lattice."""
    l = links.shape[0]
    acc = 0.0 + 0.0j

    for x in range(l):
        for y in range(l):
            phase = loop_phase_at_start(links, x=x, y=y, r=r, t=t)
            acc += np.exp(1j * phase)

    return acc / (l * l)


def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sem


def build_loop_report(loop_samples: dict[tuple[int, int], list[complex]]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for (r, t), samples in loop_samples.items():
        arr = np.asarray(samples, dtype=np.complex128)
        real_mean, real_sem = mean_and_sem(arr.real)
        imag_mean, imag_sem = mean_and_sem(arr.imag)
        area = r * t
        sigma_local = -np.log(np.clip(real_mean, 1e-12, None)) / area

        rows.append(
            {
                "R": r,
                "T": t,
                "area": area,
                "W_real_mean": real_mean,
                "W_real_sem": real_sem,
                "W_imag_mean": imag_mean,
                "W_imag_sem": imag_sem,
                "sigma_local": float(sigma_local),
            }
        )

    return pd.DataFrame(rows).sort_values(["area", "R", "T"]).reset_index(drop=True)


def estimate_string_tension(loop_df: pd.DataFrame) -> tuple[float, float]:
    areas = loop_df["area"].to_numpy(dtype=float)
    y = -np.log(np.clip(loop_df["W_real_mean"].to_numpy(dtype=float), 1e-12, None))

    denom = float(np.dot(areas, areas))
    if denom <= 0.0:
        raise ValueError("Invalid loop areas for string tension fit.")

    sigma = float(np.dot(areas, y) / denom)
    y_hat = sigma * areas

    residual = float(np.sum((y - y_hat) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 1e-15 else 1.0
    return sigma, r2


def creutz_ratio(loop_means: dict[tuple[int, int], float], r: int, t: int) -> float:
    """Compute chi(R,T) = -ln(W(R,T) W(R-1,T-1) / (W(R,T-1) W(R-1,T)))."""
    required = [(r, t), (r - 1, t - 1), (r, t - 1), (r - 1, t)]
    for key in required:
        if key not in loop_means:
            raise KeyError(f"Missing loop for Creutz ratio: {key}")

    numerator = loop_means[(r, t)] * loop_means[(r - 1, t - 1)]
    denominator = loop_means[(r, t - 1)] * loop_means[(r - 1, t)]
    ratio = np.clip(numerator / max(denominator, 1e-12), 1e-12, None)
    return float(-np.log(ratio))


def run_quality_checks(
    thermal_acceptance: float,
    measurement_acceptance: float,
    plaquette_samples: np.ndarray,
    loop_df: pd.DataFrame,
) -> None:
    if not np.isfinite(thermal_acceptance) or not np.isfinite(measurement_acceptance):
        raise AssertionError("Acceptance rates must be finite.")
    if not (0.05 < thermal_acceptance < 0.95):
        raise AssertionError(f"Thermal acceptance out of reasonable range: {thermal_acceptance:.3f}")
    if not (0.05 < measurement_acceptance < 0.95):
        raise AssertionError(f"Measurement acceptance out of reasonable range: {measurement_acceptance:.3f}")

    plaquette_mean = float(np.mean(plaquette_samples))
    if not (-1.0 <= plaquette_mean <= 1.0):
        raise AssertionError("Average plaquette must lie in [-1, 1].")

    if not np.all(np.isfinite(loop_df["W_real_mean"])):
        raise AssertionError("Wilson loop real parts contain non-finite values.")
    if not np.all(np.isfinite(loop_df["W_real_sem"])):
        raise AssertionError("Wilson loop errors contain non-finite values.")

    loop_map = {(int(r), int(t)): float(w) for r, t, w in loop_df[["R", "T", "W_real_mean"]].itertuples(index=False, name=None)}
    if (1, 1) in loop_map and (2, 2) in loop_map:
        if not (loop_map[(1, 1)] > loop_map[(2, 2)]):
            raise AssertionError("Expected W(1,1) > W(2,2) for this area-law style regime.")

    max_imag = float(np.max(np.abs(loop_df["W_imag_mean"].to_numpy(dtype=float))))
    if max_imag > 0.08:
        raise AssertionError(f"Wilson loop imaginary contamination too high: {max_imag:.4f}")


def main() -> None:
    config = WilsonLoopConfig()
    validate_config(config)

    rng = np.random.default_rng(config.seed)
    l = config.lattice_size
    links = rng.uniform(-np.pi, np.pi, size=(l, l, 2))

    thermal_accepted = 0
    thermal_total = 0
    for _ in range(config.thermalization_sweeps):
        a, t = metropolis_sweep(
            links=links,
            beta=config.beta,
            proposal_width=config.proposal_width,
            rng=rng,
        )
        thermal_accepted += a
        thermal_total += t

    loop_samples: dict[tuple[int, int], list[complex]] = {shape: [] for shape in config.loop_shapes}
    plaquette_samples: list[float] = []

    meas_accepted = 0
    meas_total = 0

    for _ in range(config.measurement_sweeps):
        for _ in range(config.sweeps_between_measurements):
            a, t = metropolis_sweep(
                links=links,
                beta=config.beta,
                proposal_width=config.proposal_width,
                rng=rng,
            )
            meas_accepted += a
            meas_total += t

        plaquette_samples.append(average_plaquette(links))
        for shape in config.loop_shapes:
            loop_samples[shape].append(measure_wilson_loop(links, r=shape[0], t=shape[1]))

    thermal_acceptance = thermal_accepted / thermal_total
    measurement_acceptance = meas_accepted / meas_total

    plaquette_arr = np.asarray(plaquette_samples, dtype=float)
    plaq_mean, plaq_sem = mean_and_sem(plaquette_arr)

    loop_df = build_loop_report(loop_samples)
    sigma_fit, sigma_r2 = estimate_string_tension(loop_df)

    loop_means = {
        shape: float(np.mean(np.asarray(values, dtype=np.complex128).real))
        for shape, values in loop_samples.items()
    }
    chi_22 = creutz_ratio(loop_means, r=2, t=2)

    print("=== 2D U(1) Wilson Loop Monte Carlo ===")
    print(
        f"L={config.lattice_size}, beta={config.beta:.3f}, proposal_width={config.proposal_width:.3f}, "
        f"thermal_sweeps={config.thermalization_sweeps}, measure_sweeps={config.measurement_sweeps}, "
        f"gap={config.sweeps_between_measurements}"
    )
    print(f"thermal_acceptance={thermal_acceptance:.4f}")
    print(f"measurement_acceptance={measurement_acceptance:.4f}")
    print(f"<plaquette>={plaq_mean:.6f} ± {plaq_sem:.6f}")
    print()
    print("=== Wilson Loop Table ===")
    print(loop_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print()
    print("=== Derived Quantities ===")
    print(f"sigma_fit(area-law linear through origin) = {sigma_fit:.6f}")
    print(f"fit_R2 = {sigma_r2:.6f}")
    print(f"Creutz ratio chi(2,2) = {chi_22:.6f}")

    run_quality_checks(
        thermal_acceptance=thermal_acceptance,
        measurement_acceptance=measurement_acceptance,
        plaquette_samples=plaquette_arr,
        loop_df=loop_df,
    )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
