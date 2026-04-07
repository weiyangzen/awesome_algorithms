"""Minimal runnable MVP for jet clustering algorithms.

This demo implements generalized sequential-recombination jet clustering:
- k_t (p=+1)
- Cambridge/Aachen (p=0)
- anti-k_t (p=-1)

It uses explicit O(N^3) scans for educational transparency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Particle:
    """Input particle in (pt, eta, phi, mass)."""

    pid: int
    pt: float
    eta: float
    phi: float
    mass: float = 0.139


@dataclass
class PseudoJet:
    """Mutable clustering object with four-momentum and constituent ids."""

    p4: np.ndarray  # [px, py, pz, E]
    constituents: List[int]


@dataclass(frozen=True)
class Jet:
    """Final reconstructed jet."""

    jet_id: str
    pt: float
    eta: float
    phi: float
    mass: float
    n_constituents: int
    constituents: Tuple[int, ...]


@dataclass(frozen=True)
class ClusterResult:
    """One algorithm run result bundle."""

    algorithm: str
    p: float
    radius: float
    pt_min: float
    jets: Tuple[Jet, ...]
    pair_merges: int
    beam_declarations: int
    iterations: int
    n_particles: int
    n_final_pseudojets_before_cut: int
    trace_table: pd.DataFrame


def wrap_phi(phi: float) -> float:
    """Wrap azimuth to [-pi, pi)."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def delta_phi(phi1: float, phi2: float) -> float:
    """Shortest signed azimuth difference."""
    return wrap_phi(phi1 - phi2)


def p4_from_pt_eta_phi_m(pt: float, eta: float, phi: float, mass: float) -> np.ndarray:
    """Convert cylindrical coordinates to Cartesian four-vector."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.array([px, py, pz, energy], dtype=float)


def kinematics_from_p4(p4: np.ndarray) -> Tuple[float, float, float, float]:
    """Recover (pt, eta, phi, mass) from [px, py, pz, E]."""
    px, py, pz, energy = p4
    pt = float(np.hypot(px, py))
    phi = float(np.arctan2(py, px))

    p_abs = float(np.sqrt(px * px + py * py + pz * pz))
    if p_abs <= abs(pz):
        eta = float(np.sign(pz) * 1.0e6)
    else:
        eta = float(0.5 * np.log((p_abs + pz) / (p_abs - pz)))

    m2 = float(energy * energy - p_abs * p_abs)
    mass = float(np.sqrt(max(m2, 0.0)))
    return pt, eta, phi, mass


def particle_to_pseudojet(particle: Particle) -> PseudoJet:
    p4 = p4_from_pt_eta_phi_m(particle.pt, particle.eta, particle.phi, particle.mass)
    return PseudoJet(p4=p4, constituents=[particle.pid])


def pt_power_weight(pt: float, p: float) -> float:
    """Return pt^(2p) with numerical safeguards.

    For p=0 (Cambridge/Aachen), this returns 1.
    """
    safe_pt = max(float(pt), 1.0e-12)
    if abs(p) < 1.0e-15:
        return 1.0
    return safe_pt ** (2.0 * p)


def generalized_pair_distance(a: PseudoJet, b: PseudoJet, radius: float, p: float) -> float:
    """Generalized-k_t pair distance.

    d_ij = min(pt_i^(2p), pt_j^(2p)) * (DeltaR_ij^2 / R^2)
    """
    pt_a, eta_a, phi_a, _ = kinematics_from_p4(a.p4)
    pt_b, eta_b, phi_b, _ = kinematics_from_p4(b.p4)

    dphi = delta_phi(phi_a, phi_b)
    deta = eta_a - eta_b
    delta_r2 = deta * deta + dphi * dphi

    prefactor = min(pt_power_weight(pt_a, p), pt_power_weight(pt_b, p))
    return prefactor * (delta_r2 / (radius * radius))


def generalized_beam_distance(pj: PseudoJet, p: float) -> float:
    """Generalized-k_t beam distance: d_iB = pt_i^(2p)."""
    pt, _, _, _ = kinematics_from_p4(pj.p4)
    return pt_power_weight(pt, p)


def merge_pseudojets(a: PseudoJet, b: PseudoJet) -> PseudoJet:
    """E-scheme recombination: add four-vectors directly."""
    return PseudoJet(p4=a.p4 + b.p4, constituents=a.constituents + b.constituents)


def generalized_kt_cluster(
    particles: Sequence[Particle],
    algorithm: str,
    p: float,
    radius: float = 0.6,
    pt_min: float = 15.0,
    max_trace_rows: int = 14,
) -> ClusterResult:
    """Cluster one event with generalized sequential recombination."""
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if pt_min < 0.0:
        raise ValueError("pt_min must be non-negative.")

    active: List[PseudoJet] = [particle_to_pseudojet(x) for x in particles]
    final_pseudojets: List[PseudoJet] = []

    pair_merges = 0
    beam_declarations = 0
    iterations = 0
    trace_rows: List[dict] = []

    while active:
        iterations += 1
        n_active_before = len(active)

        best_distance = float("inf")
        best_action = "beam"
        best_i = -1
        best_j = -1

        for i, pj in enumerate(active):
            dij_beam = generalized_beam_distance(pj, p=p)
            if dij_beam < best_distance:
                best_distance = dij_beam
                best_action = "beam"
                best_i = i
                best_j = -1

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dij_pair = generalized_pair_distance(active[i], active[j], radius=radius, p=p)
                if dij_pair < best_distance:
                    best_distance = dij_pair
                    best_action = "pair"
                    best_i = i
                    best_j = j

        if len(trace_rows) < max_trace_rows:
            trace_rows.append(
                {
                    "iter": iterations,
                    "n_active_before": n_active_before,
                    "action": best_action,
                    "best_distance": best_distance,
                    "i": best_i,
                    "j": best_j,
                }
            )

        if best_action == "pair":
            pair_merges += 1
            i, j = sorted((best_i, best_j))
            pj_j = active.pop(j)
            pj_i = active.pop(i)
            active.append(merge_pseudojets(pj_i, pj_j))
        else:
            beam_declarations += 1
            final_pseudojets.append(active.pop(best_i))

    # Internal consistency checks for the clustering chain.
    if iterations != len(particles):
        raise RuntimeError(
            f"Unexpected iteration count: iterations={iterations}, n_particles={len(particles)}"
        )
    if pair_merges + beam_declarations != iterations:
        raise RuntimeError("pair/beam accounting mismatch.")

    all_constituents: List[int] = []
    for pj in final_pseudojets:
        all_constituents.extend(pj.constituents)
    if sorted(all_constituents) != sorted(p.pid for p in particles):
        raise RuntimeError("Constituent bookkeeping mismatch after clustering.")

    jets_unsorted: List[Jet] = []
    for idx, pj in enumerate(final_pseudojets):
        pt, eta, phi, mass = kinematics_from_p4(pj.p4)
        if pt < pt_min:
            continue
        jets_unsorted.append(
            Jet(
                jet_id=f"J{idx}",
                pt=pt,
                eta=eta,
                phi=phi,
                mass=mass,
                n_constituents=len(pj.constituents),
                constituents=tuple(sorted(pj.constituents)),
            )
        )

    jets_sorted = sorted(jets_unsorted, key=lambda x: x.pt, reverse=True)
    trace_df = pd.DataFrame(trace_rows)

    return ClusterResult(
        algorithm=algorithm,
        p=p,
        radius=radius,
        pt_min=pt_min,
        jets=tuple(jets_sorted),
        pair_merges=pair_merges,
        beam_declarations=beam_declarations,
        iterations=iterations,
        n_particles=len(particles),
        n_final_pseudojets_before_cut=len(final_pseudojets),
        trace_table=trace_df,
    )


def generate_synthetic_event(seed: int = 21) -> List[Particle]:
    """Create a deterministic toy event with hard cores + soft background."""
    rng = np.random.default_rng(seed)

    particles: List[Particle] = []
    next_pid = 0

    def add_cluster(
        n: int,
        pt_base: float,
        pt_scale: float,
        eta_center: float,
        phi_center: float,
        eta_sigma: float,
        phi_sigma: float,
    ) -> None:
        nonlocal next_pid
        for _ in range(n):
            pt = pt_base + rng.exponential(pt_scale)
            eta = rng.normal(eta_center, eta_sigma)
            phi = wrap_phi(rng.normal(phi_center, phi_sigma))
            particles.append(
                Particle(
                    pid=next_pid,
                    pt=float(pt),
                    eta=float(eta),
                    phi=float(phi),
                )
            )
            next_pid += 1

    # Two hard structures and one medium structure.
    add_cluster(30, 5.0, 9.0, 0.85, 0.55, 0.12, 0.12)
    add_cluster(28, 4.8, 8.0, -0.72, -2.18, 0.14, 0.14)
    add_cluster(16, 2.0, 4.2, 0.10, 2.35, 0.18, 0.18)

    # Diffuse soft background.
    for _ in range(36):
        particles.append(
            Particle(
                pid=next_pid,
                pt=float(0.25 + rng.exponential(1.6)),
                eta=float(rng.uniform(-3.2, 3.2)),
                phi=float(rng.uniform(-np.pi, np.pi)),
            )
        )
        next_pid += 1

    return particles


def jets_to_frame(jets: Sequence[Jet], top_n: int = 5) -> pd.DataFrame:
    rows: List[dict] = []
    for rank, jet in enumerate(jets[:top_n], start=1):
        rows.append(
            {
                "rank": rank,
                "jet_id": jet.jet_id,
                "pt": jet.pt,
                "eta": jet.eta,
                "phi": jet.phi,
                "mass": jet.mass,
                "n_const": jet.n_constituents,
            }
        )
    return pd.DataFrame(rows)


def run_algorithm_suite(
    particles: Sequence[Particle],
    radius: float,
    pt_min: float,
) -> Tuple[List[ClusterResult], pd.DataFrame]:
    algo_specs = [
        ("kt", 1.0),
        ("cambridge_aachen", 0.0),
        ("anti_kt", -1.0),
    ]

    results: List[ClusterResult] = []
    summary_rows: List[dict] = []

    for name, p in algo_specs:
        result = generalized_kt_cluster(
            particles=particles,
            algorithm=name,
            p=p,
            radius=radius,
            pt_min=pt_min,
            max_trace_rows=12,
        )
        results.append(result)

        lead_pt = float(result.jets[0].pt) if result.jets else 0.0
        summary_rows.append(
            {
                "algorithm": name,
                "p": p,
                "n_jets_pt_ge_cut": len(result.jets),
                "leading_pt": lead_pt,
                "pair_merges": result.pair_merges,
                "beam_declarations": result.beam_declarations,
                "iterations": result.iterations,
                "n_particles": result.n_particles,
            }
        )

    return results, pd.DataFrame(summary_rows)


def main() -> None:
    particles = generate_synthetic_event(seed=21)
    radius = 0.6
    pt_min = 15.0

    results, summary_df = run_algorithm_suite(
        particles=particles,
        radius=radius,
        pt_min=pt_min,
    )

    print("=== Jet Clustering Algorithms MVP ===")
    print(f"input particles: {len(particles)}")
    print(f"radius R: {radius:.2f}, pt_min: {pt_min:.1f}")
    print()

    for result in results:
        print(f"--- Algorithm: {result.algorithm} (p={result.p:+.1f}) ---")
        print(
            f"jets above cut: {len(result.jets)}, "
            f"pair_merges: {result.pair_merges}, beam_declarations: {result.beam_declarations}"
        )

        jet_df = jets_to_frame(result.jets, top_n=4)
        if jet_df.empty:
            print("(no jets above pt cut)")
        else:
            print(jet_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if not result.trace_table.empty:
            print("trace (first clustering decisions):")
            print(
                result.trace_table.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6e}",
                )
            )
        print()

    print("=== Summary ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Deterministic sanity checks for this synthetic event.
    assert (summary_df["n_jets_pt_ge_cut"] >= 2).all(), "Each algorithm should find at least 2 hard jets."
    assert (summary_df["leading_pt"] > pt_min).all(), "Leading jet must pass pt cut."
    assert (summary_df["iterations"] == summary_df["n_particles"]).all(), "Each loop should consume one active object."

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
