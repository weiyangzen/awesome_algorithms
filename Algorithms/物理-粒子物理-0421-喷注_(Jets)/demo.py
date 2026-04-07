"""Minimal runnable MVP for anti-kt jet clustering.

This script generates a synthetic particle-level event and clusters jets
with a straightforward O(N^3) anti-kt implementation using E-scheme
four-momentum recombination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Particle:
    """Input particle in cylindrical coordinates."""

    pid: int
    pt: float
    eta: float
    phi: float
    mass: float = 0.139  # charged pion mass scale in GeV (toy default)


@dataclass
class PseudoJet:
    """Mutable clustering object with four-momentum and constituents."""

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


def wrap_phi(phi: float) -> float:
    """Wrap azimuth into [-pi, pi)."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def delta_phi(phi1: float, phi2: float) -> float:
    """Shortest signed azimuth difference."""
    return wrap_phi(phi1 - phi2)


def p4_from_pt_eta_phi_m(pt: float, eta: float, phi: float, mass: float) -> np.ndarray:
    """Convert (pt, eta, phi, m) to Cartesian four-vector [px, py, pz, E]."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.array([px, py, pz, energy], dtype=float)


def kinematics_from_p4(p4: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (pt, eta, phi, mass) from Cartesian four-vector."""
    px, py, pz, energy = p4
    pt = float(np.hypot(px, py))
    phi = float(np.arctan2(py, px))

    p_abs = float(np.sqrt(px * px + py * py + pz * pz))
    # Numerical guard near lightlike kinematics.
    if p_abs <= abs(pz):
        eta = float(np.sign(pz) * 1.0e6)
    else:
        eta = float(0.5 * np.log((p_abs + pz) / (p_abs - pz)))

    m2 = float(energy * energy - p_abs * p_abs)
    mass = float(np.sqrt(max(m2, 0.0)))
    return pt, eta, phi, mass


def particle_to_pseudojet(particle: Particle) -> PseudoJet:
    """Create initial pseudojet from one particle."""
    p4 = p4_from_pt_eta_phi_m(particle.pt, particle.eta, particle.phi, particle.mass)
    return PseudoJet(p4=p4, constituents=[particle.pid])


def anti_kt_pair_distance(a: PseudoJet, b: PseudoJet, radius: float) -> float:
    """Compute anti-kt pair distance d_ij."""
    pt_a, eta_a, phi_a, _ = kinematics_from_p4(a.p4)
    pt_b, eta_b, phi_b, _ = kinematics_from_p4(b.p4)

    safe_pt_a = max(pt_a, 1.0e-12)
    safe_pt_b = max(pt_b, 1.0e-12)

    dphi = delta_phi(phi_a, phi_b)
    deta = eta_a - eta_b
    delta_r2 = deta * deta + dphi * dphi
    inv_pt2 = min(1.0 / (safe_pt_a * safe_pt_a), 1.0 / (safe_pt_b * safe_pt_b))
    return inv_pt2 * (delta_r2 / (radius * radius))


def beam_distance(pj: PseudoJet) -> float:
    """Compute anti-kt beam distance d_iB."""
    pt, _, _, _ = kinematics_from_p4(pj.p4)
    safe_pt = max(pt, 1.0e-12)
    return 1.0 / (safe_pt * safe_pt)


def merge_pseudojets(a: PseudoJet, b: PseudoJet) -> PseudoJet:
    """E-scheme recombination: four-vectors add linearly."""
    merged_p4 = a.p4 + b.p4
    merged_constituents = a.constituents + b.constituents
    return PseudoJet(p4=merged_p4, constituents=merged_constituents)


def anti_kt_cluster(
    particles: Sequence[Particle],
    radius: float = 0.6,
    pt_min: float = 10.0,
) -> List[Jet]:
    """Cluster particles with anti-kt and return jets above pt_min."""
    active: List[PseudoJet] = [particle_to_pseudojet(p) for p in particles]
    final_pseudojets: List[PseudoJet] = []

    while active:
        best_distance = float("inf")
        best_action = "beam"
        best_i = -1
        best_j = -1

        for i, pj in enumerate(active):
            dij_beam = beam_distance(pj)
            if dij_beam < best_distance:
                best_distance = dij_beam
                best_action = "beam"
                best_i = i
                best_j = -1

        n_active = len(active)
        for i in range(n_active):
            for j in range(i + 1, n_active):
                dij_pair = anti_kt_pair_distance(active[i], active[j], radius)
                if dij_pair < best_distance:
                    best_distance = dij_pair
                    best_action = "pair"
                    best_i = i
                    best_j = j

        if best_action == "pair":
            i, j = sorted((best_i, best_j), reverse=True)
            pj_i = active.pop(i)
            pj_j = active.pop(j)
            active.append(merge_pseudojets(pj_i, pj_j))
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


def generate_synthetic_event(seed: int = 7) -> List[Particle]:
    """Generate a toy event with two hard jet cores and soft background."""
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
            particles.append(Particle(pid=next_pid, pt=float(pt), eta=float(eta), phi=float(phi)))
            next_pid += 1

    # Two hard cores.
    add_cluster(
        n=30,
        pt_base=4.0,
        pt_scale=8.0,
        eta_center=0.75,
        phi_center=0.55,
        eta_sigma=0.12,
        phi_sigma=0.12,
    )
    add_cluster(
        n=28,
        pt_base=4.5,
        pt_scale=7.0,
        eta_center=-0.65,
        phi_center=-2.10,
        eta_sigma=0.14,
        phi_sigma=0.14,
    )

    # Diffuse soft background.
    for _ in range(38):
        pt = 0.3 + rng.exponential(1.8)
        eta = rng.uniform(-3.0, 3.0)
        phi = rng.uniform(-np.pi, np.pi)
        particles.append(Particle(pid=next_pid, pt=float(pt), eta=float(eta), phi=float(phi)))
        next_pid += 1

    return particles


def format_jets_table(jets: Sequence[Jet]) -> str:
    """Format a compact plain-text table."""
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
    """Run a deterministic anti-kt clustering demo."""
    particles = generate_synthetic_event(seed=7)
    jets = anti_kt_cluster(particles, radius=0.6, pt_min=10.0)

    print(f"Generated particles: {len(particles)}")
    print(f"Clustered jets (pt >= 10.0): {len(jets)}")
    print(format_jets_table(jets))

    if jets:
        lead = jets[0]
        print(
            "Leading jet constituents (first 20 ids): "
            + ", ".join(str(x) for x in lead.constituents[:20])
        )


if __name__ == "__main__":
    main()
