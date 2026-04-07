"""Minimal runnable MVP for the Anti-kT jet clustering algorithm.

The script generates a deterministic toy event and runs a straightforward
O(N^3) anti-kT sequential recombination implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Particle:
    """Input particle represented in cylindrical coordinates."""

    pid: int
    pt: float
    eta: float
    phi: float
    mass: float = 0.13957


@dataclass
class PseudoJet:
    """Clustering state object with four-momentum and constituent ids."""

    p4: np.ndarray  # [px, py, pz, E]
    constituents: List[int]


@dataclass(frozen=True)
class Jet:
    """Final reconstructed jet object."""

    jet_id: str
    pt: float
    eta: float
    phi: float
    mass: float
    n_constituents: int
    constituents: Tuple[int, ...]


def wrap_phi(phi: float) -> float:
    """Wrap azimuth into [-pi, pi)."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def delta_phi(phi1: float, phi2: float) -> float:
    """Return shortest signed azimuth difference."""
    return wrap_phi(phi1 - phi2)


def p4_from_pt_eta_phi_m(pt: float, eta: float, phi: float, mass: float) -> np.ndarray:
    """Convert (pt, eta, phi, m) to [px, py, pz, E]."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.array([px, py, pz, energy], dtype=float)


def kinematics_from_p4(p4: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert [px, py, pz, E] back to (pt, eta, phi, m)."""
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
    """Create one initial pseudojet from one particle."""
    p4 = p4_from_pt_eta_phi_m(particle.pt, particle.eta, particle.phi, particle.mass)
    return PseudoJet(p4=p4, constituents=[particle.pid])


def beam_distance(pj: PseudoJet) -> float:
    """Compute anti-kT beam distance d_iB = 1/pt_i^2."""
    pt, _, _, _ = kinematics_from_p4(pj.p4)
    safe_pt = max(pt, 1.0e-12)
    return 1.0 / (safe_pt * safe_pt)


def anti_kt_pair_distance(a: PseudoJet, b: PseudoJet, radius: float) -> float:
    """Compute anti-kT pair distance d_ij."""
    pt_a, eta_a, phi_a, _ = kinematics_from_p4(a.p4)
    pt_b, eta_b, phi_b, _ = kinematics_from_p4(b.p4)

    safe_pt_a = max(pt_a, 1.0e-12)
    safe_pt_b = max(pt_b, 1.0e-12)

    deta = eta_a - eta_b
    dphi = delta_phi(phi_a, phi_b)
    delta_r2 = deta * deta + dphi * dphi

    inv_pt2 = min(1.0 / (safe_pt_a * safe_pt_a), 1.0 / (safe_pt_b * safe_pt_b))
    return inv_pt2 * (delta_r2 / (radius * radius))


def merge_pseudojets(a: PseudoJet, b: PseudoJet) -> PseudoJet:
    """E-scheme recombination by four-vector addition."""
    return PseudoJet(p4=a.p4 + b.p4, constituents=a.constituents + b.constituents)


def anti_kt_cluster(
    particles: Sequence[Particle],
    radius: float = 0.6,
    pt_min: float = 8.0,
) -> List[Jet]:
    """Cluster a particle list into jets using a minimal anti-kT implementation."""
    active: List[PseudoJet] = [particle_to_pseudojet(p) for p in particles]
    final_pseudojets: List[PseudoJet] = []

    while active:
        best_distance = float("inf")
        best_mode = "beam"
        best_i = -1
        best_j = -1

        for i, pj in enumerate(active):
            d_iB = beam_distance(pj)
            if d_iB < best_distance:
                best_distance = d_iB
                best_mode = "beam"
                best_i = i
                best_j = -1

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                d_ij = anti_kt_pair_distance(active[i], active[j], radius)
                if d_ij < best_distance:
                    best_distance = d_ij
                    best_mode = "pair"
                    best_i = i
                    best_j = j

        if best_mode == "pair":
            high = max(best_i, best_j)
            low = min(best_i, best_j)
            pj_high = active.pop(high)
            pj_low = active.pop(low)
            active.append(merge_pseudojets(pj_high, pj_low))
        else:
            final_pseudojets.append(active.pop(best_i))

    jets: List[Jet] = []
    for idx, pj in enumerate(final_pseudojets):
        pt, eta, phi, mass = kinematics_from_p4(pj.p4)
        if pt < pt_min:
            continue
        jets.append(
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

    jets.sort(key=lambda x: x.pt, reverse=True)
    return jets


def generate_toy_event(seed: int = 424) -> List[Particle]:
    """Create a deterministic toy event with two hard cores plus soft UE-like background."""
    rng = np.random.default_rng(seed)

    particles: List[Particle] = []
    next_pid = 0

    def add_core(
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

    add_core(
        n=30,
        pt_base=4.0,
        pt_scale=7.0,
        eta_center=0.55,
        phi_center=0.78,
        eta_sigma=0.13,
        phi_sigma=0.13,
    )
    add_core(
        n=28,
        pt_base=4.5,
        pt_scale=6.5,
        eta_center=-0.82,
        phi_center=-2.26,
        eta_sigma=0.14,
        phi_sigma=0.14,
    )

    for _ in range(32):
        pt = 0.25 + rng.exponential(1.6)
        eta = rng.uniform(-3.2, 3.2)
        phi = rng.uniform(-np.pi, np.pi)
        particles.append(
            Particle(
                pid=next_pid,
                pt=float(pt),
                eta=float(eta),
                phi=float(phi),
            )
        )
        next_pid += 1

    return particles


def format_jets_table(jets: Sequence[Jet]) -> str:
    """Format jets as a compact plain-text table."""
    if not jets:
        return "(no jets above threshold)"

    header = f"{'jet_id':<6} {'pt':>9} {'eta':>8} {'phi':>8} {'mass':>9} {'n_const':>8}"
    lines = [header, "-" * len(header)]

    for jet in jets:
        lines.append(
            f"{jet.jet_id:<6} {jet.pt:9.3f} {jet.eta:8.3f} {jet.phi:8.3f} {jet.mass:9.3f} {jet.n_constituents:8d}"
        )

    return "\n".join(lines)


def main() -> None:
    """Run a non-interactive anti-kT demo."""
    particles = generate_toy_event(seed=424)
    jets = anti_kt_cluster(particles, radius=0.6, pt_min=8.0)

    print(f"Generated particles: {len(particles)}")
    print(f"Clustered jets (pt >= 8.0): {len(jets)}")
    print(format_jets_table(jets))

    if jets:
        lead = jets[0]
        preview = ", ".join(str(x) for x in lead.constituents[:20])
        print(f"Leading jet first 20 constituent ids: {preview}")


if __name__ == "__main__":
    main()
