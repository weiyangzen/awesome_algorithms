"""Minimal runnable MVP for Markov Random Field (MATH-0325).

This demo builds a binary Ising MRF denoising task on a 2D grid:
1) synthesize a structured ground-truth field in {-1, +1};
2) corrupt it with Bernoulli flip noise;
3) run ICM (Iterated Conditional Modes) for MAP inference.

No black-box MRF library is used. Energy, local updates, and validations are
implemented directly for source-level traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


SpinGrid = np.ndarray


@dataclass(frozen=True)
class MRFConfig:
    """Configuration for one reproducible MRF denoising run."""

    height: int = 28
    width: int = 28
    noise_flip_prob: float = 0.18
    beta: float = 1.1
    eta: float = 1.6
    max_iters: int = 25
    seed: int = 325


def generate_ground_truth(height: int, width: int) -> SpinGrid:
    """Generate a structured binary field in {-1, +1}."""
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    truth = np.where(radius < min(height, width) * 0.24, 1, -1).astype(np.int8)

    # Add two extra regions so denoising is not trivial.
    truth[(yy > height * 0.65) & (xx < width * 0.35)] = 1
    truth[(yy < height * 0.25) & (xx > width * 0.60)] = 1
    return truth


def add_bernoulli_flip_noise(
    truth: SpinGrid,
    flip_prob: float,
    rng: np.random.Generator,
) -> SpinGrid:
    """Flip each site independently with probability flip_prob."""
    flips = rng.random(truth.shape) < flip_prob
    obs = truth.copy()
    obs[flips] *= -1
    return obs


def local_neighbor_sum(x: SpinGrid, i: int, j: int) -> int:
    """4-neighborhood sum for one lattice site."""
    h, w = x.shape
    s = 0
    if i > 0:
        s += int(x[i - 1, j])
    if i + 1 < h:
        s += int(x[i + 1, j])
    if j > 0:
        s += int(x[i, j - 1])
    if j + 1 < w:
        s += int(x[i, j + 1])
    return s


def ising_posterior_energy(x: SpinGrid, y: SpinGrid, beta: float, eta: float) -> float:
    """Posterior energy E(x|y) up to additive constant.

    E(x|y) = -eta * sum_i (x_i y_i)
             -beta * sum_(i,j) neighbors (x_i x_j)
    """
    unary = -eta * float(np.sum(x * y))
    pair_h = float(np.sum(x[:-1, :] * x[1:, :]))
    pair_w = float(np.sum(x[:, :-1] * x[:, 1:]))
    pairwise = -beta * (pair_h + pair_w)
    return unary + pairwise


def icm_map_inference(
    y: SpinGrid,
    beta: float,
    eta: float,
    max_iters: int,
) -> Tuple[SpinGrid, List[float], List[int]]:
    """ICM MAP inference with red-black (checkerboard) updates."""
    h, w = y.shape
    x = y.copy()

    energies: List[float] = []
    changed_counts: List[int] = []

    for _ in range(max_iters):
        changed = 0

        for parity in (0, 1):
            for i in range(h):
                for j in range(w):
                    if ((i + j) & 1) != parity:
                        continue

                    field = eta * int(y[i, j]) + beta * local_neighbor_sum(x, i, j)
                    new_spin = 1 if field >= 0.0 else -1
                    if new_spin != int(x[i, j]):
                        x[i, j] = np.int8(new_spin)
                        changed += 1

        energies.append(ising_posterior_energy(x=x, y=y, beta=beta, eta=eta))
        changed_counts.append(changed)
        if changed == 0:
            break

    return x, energies, changed_counts


def pixel_accuracy(a: SpinGrid, b: SpinGrid) -> float:
    """Fraction of equal spins."""
    return float(np.mean(a == b))


def boundary_disagreement_count(x: SpinGrid) -> int:
    """Count neighboring pairs with different labels (a smoothness proxy)."""
    vertical = int(np.sum(x[:-1, :] != x[1:, :]))
    horizontal = int(np.sum(x[:, :-1] != x[:, 1:]))
    return vertical + horizontal


def monotone_non_increasing(values: List[float], tol: float = 1e-12) -> bool:
    """Check monotonic non-increase within tolerance."""
    return all(values[i + 1] <= values[i] + tol for i in range(len(values) - 1))


def to_ascii(grid: SpinGrid, rows: int = 12, pos: str = "#", neg: str = ".") -> str:
    """Render top rows of a spin grid for quick terminal inspection."""
    n_rows = min(rows, grid.shape[0])
    lines: List[str] = []
    for i in range(n_rows):
        chars = [pos if int(v) > 0 else neg for v in grid[i]]
        lines.append("".join(chars))
    return "\n".join(lines)


def main() -> None:
    config = MRFConfig()
    rng = np.random.default_rng(config.seed)

    x_true = generate_ground_truth(height=config.height, width=config.width)
    y_obs = add_bernoulli_flip_noise(
        truth=x_true,
        flip_prob=config.noise_flip_prob,
        rng=rng,
    )

    initial_energy = ising_posterior_energy(x=y_obs, y=y_obs, beta=config.beta, eta=config.eta)
    x_map, energies, changed_counts = icm_map_inference(
        y=y_obs,
        beta=config.beta,
        eta=config.eta,
        max_iters=config.max_iters,
    )

    noisy_acc = pixel_accuracy(y_obs, x_true)
    map_acc = pixel_accuracy(x_map, x_true)

    noisy_disagree = boundary_disagreement_count(y_obs)
    map_disagree = boundary_disagreement_count(x_map)

    print("Markov Random Field MVP (MATH-0325)")
    print("=" * 72)
    print(
        f"grid={config.height}x{config.width}, noise_flip_prob={config.noise_flip_prob:.2f}, "
        f"beta={config.beta:.2f}, eta={config.eta:.2f}, seed={config.seed}"
    )
    print("-" * 72)
    print(f"accuracy noisy -> MAP: {noisy_acc:.4f} -> {map_acc:.4f} (delta={map_acc - noisy_acc:+.4f})")
    print(f"boundary disagreement noisy -> MAP: {noisy_disagree} -> {map_disagree}")
    print(f"posterior energy initial(noisy): {initial_energy:.4f}")
    print(f"posterior energy final(MAP):     {energies[-1]:.4f}")
    print(f"ICM sweeps: {len(energies)}, per-sweep changed pixels: {changed_counts}")

    print("-" * 72)
    print("ground-truth preview (#=+1, .=-1):")
    print(to_ascii(x_true, rows=12))
    print("-" * 72)
    print("noisy-observation preview:")
    print(to_ascii(y_obs, rows=12))
    print("-" * 72)
    print("MAP-estimate preview:")
    print(to_ascii(x_map, rows=12))

    # Basic validation gates for this MVP run.
    assert len(energies) >= 1, "Energy history must not be empty."
    assert monotone_non_increasing(energies), "ICM energy should be monotone non-increasing."
    assert energies[-1] <= initial_energy + 1e-9, "Final energy should not exceed initial noisy energy."
    assert map_acc >= noisy_acc + 0.05, "MAP denoising should significantly improve accuracy."

    labels = set(np.unique(x_map).tolist())
    assert labels.issubset({-1, 1}), "Output labels must stay in {-1, +1}."


if __name__ == "__main__":
    main()
