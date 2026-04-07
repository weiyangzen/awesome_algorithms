"""Minimal runnable MVP for Missing Transverse Energy (MET).

The script builds a toy particle-physics event with one invisible neutrino,
reconstructs visible objects with simplified detector effects, and computes
MET from reconstructed visible transverse momentum balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TruthParticle:
    """Generator-level particle used in the toy event."""

    name: str
    kind: str
    px: float
    py: float
    eta: float
    visible: bool


@dataclass(frozen=True)
class RecoObject:
    """Reconstructed visible object used for MET calculation."""

    name: str
    kind: str
    px: float
    py: float
    eta: float


@dataclass(frozen=True)
class METResult:
    """Container for MET vector and magnitude."""

    met_x: float
    met_y: float
    met: float
    phi: float


def wrap_phi(phi: float) -> float:
    """Wrap angle into [-pi, pi)."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def delta_phi(phi1: float, phi2: float) -> float:
    """Shortest signed azimuth difference."""
    return wrap_phi(phi1 - phi2)


def px_py_to_pt_phi(px: float, py: float) -> Tuple[float, float]:
    """Convert Cartesian transverse components to (pt, phi)."""
    pt = float(np.hypot(px, py))
    phi = float(np.arctan2(py, px))
    return pt, phi


def pt_phi_to_px_py(pt: float, phi: float) -> Tuple[float, float]:
    """Convert (pt, phi) to Cartesian transverse components."""
    px = float(pt * np.cos(phi))
    py = float(pt * np.sin(phi))
    return px, py


def compute_met_from_visible(objects: Sequence[RecoObject]) -> METResult:
    """MET = -sum of visible transverse momentum vectors."""
    sum_px = float(np.sum([obj.px for obj in objects], dtype=float))
    sum_py = float(np.sum([obj.py for obj in objects], dtype=float))
    met_x = -sum_px
    met_y = -sum_py
    met = float(np.hypot(met_x, met_y))
    phi = float(np.arctan2(met_y, met_x))
    return METResult(met_x=met_x, met_y=met_y, met=met, phi=phi)


def compute_met_from_truth_visible(particles: Sequence[TruthParticle]) -> METResult:
    """Reference MET built from generator-level visible particles."""
    visible = [
        RecoObject(name=p.name, kind=p.kind, px=p.px, py=p.py, eta=p.eta)
        for p in particles
        if p.visible
    ]
    return compute_met_from_visible(visible)


def compute_truth_invisible_vector(particles: Sequence[TruthParticle]) -> METResult:
    """Truth invisible transverse momentum sum (e.g., neutrinos)."""
    inv_px = float(np.sum([p.px for p in particles if not p.visible], dtype=float))
    inv_py = float(np.sum([p.py for p in particles if not p.visible], dtype=float))
    inv_pt = float(np.hypot(inv_px, inv_py))
    inv_phi = float(np.arctan2(inv_py, inv_px))
    return METResult(met_x=inv_px, met_y=inv_py, met=inv_pt, phi=inv_phi)


def _split_vector_into_soft_terms(
    total_px: float,
    total_py: float,
    n_terms: int,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """Split one target vector into n components; last term closes momentum."""
    terms: List[Tuple[float, float]] = []
    remain_px = total_px
    remain_py = total_py
    if n_terms <= 1:
        return [(remain_px, remain_py)]

    for _ in range(n_terms - 1):
        frac = float(rng.uniform(0.06, 0.20))
        pt_remain = float(np.hypot(remain_px, remain_py))
        phi_remain = float(np.arctan2(remain_py, remain_px))

        term_pt = frac * pt_remain
        term_phi = wrap_phi(phi_remain + float(rng.normal(0.0, 1.1)))
        px, py = pt_phi_to_px_py(term_pt, term_phi)
        terms.append((px, py))
        remain_px -= px
        remain_py -= py

    terms.append((remain_px, remain_py))
    return terms


def generate_truth_event(seed: int = 425) -> List[TruthParticle]:
    """Create a toy event with one invisible neutrino and visible recoil.

    We enforce approximate transverse momentum closure at truth level:
    pT(neutrino) + pT(visible) ~= 0
    """
    rng = np.random.default_rng(seed)

    # Invisible neutrino.
    nu_pt = float(rng.uniform(35.0, 110.0))
    nu_phi = float(rng.uniform(-np.pi, np.pi))
    nu_px, nu_py = pt_phi_to_px_py(nu_pt, nu_phi)

    # Visible muon from W-like topology.
    mu_pt = float(rng.uniform(25.0, 75.0))
    mu_phi = wrap_phi(nu_phi + np.pi + float(rng.normal(0.0, 0.45)))
    mu_px, mu_py = pt_phi_to_px_py(mu_pt, mu_phi)
    mu_eta = float(rng.normal(0.0, 1.0))

    # Small diffuse soft recoil.
    soft_tot_px = float(rng.normal(0.0, 6.0))
    soft_tot_py = float(rng.normal(0.0, 6.0))

    # Hadronic recoil required by momentum balance.
    jet_tot_px = -(nu_px + mu_px + soft_tot_px)
    jet_tot_py = -(nu_py + mu_py + soft_tot_py)

    jet_vectors = _split_vector_into_soft_terms(
        total_px=jet_tot_px,
        total_py=jet_tot_py,
        n_terms=3,
        rng=rng,
    )
    soft_vectors = _split_vector_into_soft_terms(
        total_px=soft_tot_px,
        total_py=soft_tot_py,
        n_terms=5,
        rng=rng,
    )

    particles: List[TruthParticle] = [
        TruthParticle(
            name="nu_mu",
            kind="neutrino",
            px=nu_px,
            py=nu_py,
            eta=float(rng.normal(0.0, 1.6)),
            visible=False,
        ),
        TruthParticle(
            name="muon",
            kind="muon",
            px=mu_px,
            py=mu_py,
            eta=mu_eta,
            visible=True,
        ),
    ]

    for i, (px, py) in enumerate(jet_vectors):
        particles.append(
            TruthParticle(
                name=f"jet_truth_{i}",
                kind="jet",
                px=px,
                py=py,
                eta=float(rng.normal(0.0, 1.4)),
                visible=True,
            )
        )

    for i, (px, py) in enumerate(soft_vectors):
        particles.append(
            TruthParticle(
                name=f"soft_truth_{i}",
                kind="soft",
                px=px,
                py=py,
                eta=float(rng.uniform(-5.2, 5.2)),
                visible=True,
            )
        )

    return particles


def reconstruct_visible_objects(
    particles: Sequence[TruthParticle],
    seed: int = 426,
) -> List[RecoObject]:
    """Apply simplified detector acceptance + resolution to visible particles."""
    rng = np.random.default_rng(seed)

    reco: List[RecoObject] = []
    for p in particles:
        if not p.visible:
            continue

        pt_true, phi_true = px_py_to_pt_phi(p.px, p.py)
        if p.kind == "muon":
            eta_max = 2.7
            pt_min = 3.0
            sigma_rel = 0.01
            sigma_phi = 0.001
        elif p.kind == "jet":
            eta_max = 4.5
            pt_min = 10.0
            sigma_rel = 0.10
            sigma_phi = 0.02
        else:  # soft terms
            eta_max = 4.9
            pt_min = 0.5
            sigma_rel = 0.25
            sigma_phi = 0.06

        if abs(p.eta) > eta_max:
            continue

        scale = max(float(rng.normal(1.0, sigma_rel)), 0.0)
        pt_reco = pt_true * scale
        if pt_reco < pt_min:
            continue

        phi_reco = wrap_phi(phi_true + float(rng.normal(0.0, sigma_phi)))
        px_reco, py_reco = pt_phi_to_px_py(pt_reco, phi_reco)

        reco.append(
            RecoObject(name=p.name.replace("_truth", "_reco"), kind=p.kind, px=px_reco, py=py_reco, eta=p.eta)
        )

    return reco


def _format_objects_table(objects: Sequence[RecoObject]) -> str:
    """Compact plain-text table for reconstructed objects."""
    if not objects:
        return "(no reconstructed objects)"

    rows = []
    for obj in objects:
        pt, phi = px_py_to_pt_phi(obj.px, obj.py)
        rows.append((obj.name, obj.kind, pt, obj.eta, phi, obj.px, obj.py))

    rows.sort(key=lambda x: x[2], reverse=True)
    header = f"{'name':<14} {'kind':<6} {'pt':>9} {'eta':>8} {'phi':>8} {'px':>10} {'py':>10}"
    lines = [header, "-" * len(header)]
    for name, kind, pt, eta, phi, px, py in rows:
        lines.append(
            f"{name:<14} {kind:<6} {pt:9.3f} {eta:8.3f} {phi:8.3f} {px:10.3f} {py:10.3f}"
        )

    return "\n".join(lines)


def main() -> None:
    """Run deterministic MET demo with toy truth and reconstruction stages."""
    truth_particles = generate_truth_event(seed=425)
    reco_objects = reconstruct_visible_objects(truth_particles, seed=426)

    met_reco = compute_met_from_visible(reco_objects)
    met_truth_visible = compute_met_from_truth_visible(truth_particles)
    met_truth_invisible = compute_truth_invisible_vector(truth_particles)

    dphi_reco_truth = abs(delta_phi(met_reco.phi, met_truth_invisible.phi))
    delta_met = met_reco.met - met_truth_invisible.met

    n_visible_truth = sum(1 for p in truth_particles if p.visible)
    n_invisible_truth = sum(1 for p in truth_particles if not p.visible)

    print("=== Missing Transverse Energy (MET) Demo ===")
    print(f"Truth particles: total={len(truth_particles)}, visible={n_visible_truth}, invisible={n_invisible_truth}")
    print(f"Reconstructed visible objects: {len(reco_objects)}")
    print()
    print(_format_objects_table(reco_objects))
    print()

    print("--- MET Summary (GeV) ---")
    print(
        "Truth invisible pT sum: "
        f"MET={met_truth_invisible.met:.3f}, phi={met_truth_invisible.phi:.3f}, "
        f"(mx,my)=({met_truth_invisible.met_x:.3f}, {met_truth_invisible.met_y:.3f})"
    )
    print(
        "Truth from visible closure: "
        f"MET={met_truth_visible.met:.3f}, phi={met_truth_visible.phi:.3f}, "
        f"(mx,my)=({met_truth_visible.met_x:.3f}, {met_truth_visible.met_y:.3f})"
    )
    print(
        "Reco MET from visible objects: "
        f"MET={met_reco.met:.3f}, phi={met_reco.phi:.3f}, "
        f"(mx,my)=({met_reco.met_x:.3f}, {met_reco.met_y:.3f})"
    )
    print(f"Reco - Truth(|MET|): {delta_met:+.3f} GeV")
    print(f"|Delta phi(reco, truth-invisible)|: {dphi_reco_truth:.3f} rad")


if __name__ == "__main__":
    main()
