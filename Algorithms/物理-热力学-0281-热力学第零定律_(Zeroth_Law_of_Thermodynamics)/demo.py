"""Minimal runnable MVP for Zeroth Law of Thermodynamics.

Idea:
- Simulate pairwise thermal contacts between bodies.
- Decide thermal equilibrium from measured heat flux threshold.
- Build equilibrium classes via union-find (equivalence relation).
- Verify transitivity (core of the zeroth law).
- Map classes to a numerical temperature scale via a calibrated thermometer.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Body:
    """Physical body with a hidden true temperature (for synthetic validation)."""

    name: str
    true_temp_k: float


@dataclass(frozen=True)
class PairMeasurement:
    """One pairwise thermal-contact experiment result."""

    body_i: str
    body_j: str
    delta_t_true_k: float
    heat_flux_w: float
    is_equilibrium_pred: bool


class UnionFind:
    """Small disjoint-set structure for equilibrium class inference."""

    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for x in self.parent:
            r = self.find(x)
            out.setdefault(r, []).append(x)
        return out


def deterministic_noise(i: int, j: int, amplitude: float) -> float:
    """Deterministic pseudo-noise so the demo is reproducible."""
    phase = 37 * (i + 1) + 17 * (j + 1)
    return amplitude * math.sin(float(phase))


def measure_pair(
    i: int,
    j: int,
    body_i: Body,
    body_j: Body,
    conductance: float,
    noise_amplitude: float,
    flux_tol: float,
) -> PairMeasurement:
    """Simulate a contact experiment and classify equilibrium."""
    delta_t = body_i.true_temp_k - body_j.true_temp_k
    heat_flux = conductance * delta_t + deterministic_noise(i, j, noise_amplitude)
    is_eq = abs(heat_flux) <= flux_tol
    return PairMeasurement(
        body_i=body_i.name,
        body_j=body_j.name,
        delta_t_true_k=delta_t,
        heat_flux_w=heat_flux,
        is_equilibrium_pred=is_eq,
    )


def run_all_pair_measurements(
    bodies: list[Body],
    conductance: float,
    noise_amplitude: float,
    flux_tol: float,
) -> list[PairMeasurement]:
    results: list[PairMeasurement] = []
    for (i, bi), (j, bj) in itertools.combinations(enumerate(bodies), 2):
        results.append(
            measure_pair(
                i=i,
                j=j,
                body_i=bi,
                body_j=bj,
                conductance=conductance,
                noise_amplitude=noise_amplitude,
                flux_tol=flux_tol,
            )
        )
    return results


def infer_equilibrium_classes(
    body_names: list[str], measurements: list[PairMeasurement]
) -> tuple[dict[str, str], list[list[str]]]:
    """Infer thermal-equilibrium classes from pairwise predictions."""
    uf = UnionFind(body_names)
    for m in measurements:
        if m.is_equilibrium_pred:
            uf.union(m.body_i, m.body_j)

    raw_groups = [sorted(g) for g in uf.groups().values()]
    groups = sorted(raw_groups, key=lambda x: x[0])

    body_to_class: dict[str, str] = {}
    for idx, members in enumerate(groups, start=1):
        cid = f"C{idx}"
        for name in members:
            body_to_class[name] = cid

    return body_to_class, groups


def verify_zeroth_transitivity(body_to_class: dict[str, str]) -> int:
    """Check: if A~C and B~C then A~B, under inferred relation."""
    names = sorted(body_to_class)
    triples_checked = 0

    def rel(x: str, y: str) -> bool:
        return body_to_class[x] == body_to_class[y]

    for c in names:
        eq_with_c = [x for x in names if x != c and rel(x, c)]
        if len(eq_with_c) < 2:
            continue
        for a, b in itertools.combinations(eq_with_c, 2):
            assert rel(a, b), f"Transitivity failed for triple ({a}, {b}, {c})"
            triples_checked += 1

    return triples_checked


def thermometer_raw_reading(temp_k: float, body_name: str) -> float:
    """Synthetic thermometer raw reading: affine transform + deterministic bias."""
    alpha = 1.015
    beta = -12.0
    phase = sum(ord(ch) for ch in body_name)
    bias = 0.04 * math.cos(phase)
    return alpha * temp_k + beta + bias


def calibrate_kelvin_from_raw(raw_value: float) -> float:
    """Recover Kelvin by two-point calibration (273.15 K and 373.15 K)."""
    # Calibration points from the same thermometer transfer function (without bias).
    raw_0c = 1.015 * 273.15 - 12.0
    raw_100c = 1.015 * 373.15 - 12.0
    slope = (373.15 - 273.15) / (raw_100c - raw_0c)
    return 273.15 + slope * (raw_value - raw_0c)


def build_pair_frame(
    measurements: list[PairMeasurement],
    conductance: float,
    flux_tol: float,
) -> pd.DataFrame:
    temp_tol = flux_tol / conductance
    rows: list[dict[str, float | str | bool]] = []
    for m in measurements:
        gt = abs(m.delta_t_true_k) <= temp_tol
        rows.append(
            {
                "body_i": m.body_i,
                "body_j": m.body_j,
                "delta_T_true(K)": m.delta_t_true_k,
                "heat_flux(W)": m.heat_flux_w,
                "eq_pred": m.is_equilibrium_pred,
                "eq_gt": gt,
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(pair_frame: pd.DataFrame) -> dict[str, float]:
    pred = pair_frame["eq_pred"].to_numpy(dtype=bool)
    gt = pair_frame["eq_gt"].to_numpy(dtype=bool)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def build_class_frame(
    groups: list[list[str]],
    body_map: dict[str, Body],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for idx, members in enumerate(groups, start=1):
        true_temps = [body_map[n].true_temp_k for n in members]
        raw_readings = [thermometer_raw_reading(body_map[n].true_temp_k, n) for n in members]
        raw_mean = float(np.mean(raw_readings))
        est_temp = calibrate_kelvin_from_raw(raw_mean)
        rows.append(
            {
                "class_id": f"C{idx}",
                "members": ",".join(members),
                "n_members": float(len(members)),
                "true_temp_mean(K)": float(np.mean(true_temps)),
                "true_temp_span(K)": float(np.max(true_temps) - np.min(true_temps)),
                "raw_reading_mean": raw_mean,
                "est_temp_from_thermometer(K)": est_temp,
                "abs_estimation_error(K)": abs(est_temp - float(np.mean(true_temps))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # Core thresholds: |heat_flux| <= flux_tol => equilibrium prediction.
    conductance = 0.20  # W/K
    flux_tol = 0.16  # W
    noise_amplitude = 0.03  # W (deterministic pseudo-noise amplitude)

    bodies = [
        Body("A", 300.0),
        Body("B", 300.3),
        Body("C", 299.8),
        Body("D", 329.9),
        Body("E", 330.4),
        Body("F", 280.0),
        Body("G", 279.6),
        Body("H", 310.0),
    ]
    body_map = {b.name: b for b in bodies}

    measurements = run_all_pair_measurements(
        bodies=bodies,
        conductance=conductance,
        noise_amplitude=noise_amplitude,
        flux_tol=flux_tol,
    )

    pair_frame = build_pair_frame(
        measurements=measurements,
        conductance=conductance,
        flux_tol=flux_tol,
    )

    body_to_class, groups = infer_equilibrium_classes(
        body_names=[b.name for b in bodies],
        measurements=measurements,
    )
    class_frame = build_class_frame(groups=groups, body_map=body_map)

    metrics = classification_metrics(pair_frame)
    triples_checked = verify_zeroth_transitivity(body_to_class)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Zeroth Law of Thermodynamics MVP")
    print("Criterion: |heat_flux| <= flux_tol -> thermal equilibrium")
    print(f"conductance={conductance:.3f} W/K, flux_tol={flux_tol:.3f} W, noise_amp={noise_amplitude:.3f} W")

    print("\nPairwise experiments:")
    print(pair_frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nInferred equilibrium classes:")
    print(class_frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nClassification metrics against synthetic ground truth:")
    metrics_frame = pd.DataFrame([metrics])
    print(metrics_frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Assertions: deterministic and physically consistent in this synthetic setup.
    assert triples_checked > 0, "No transitivity triples checked."
    assert metrics["precision"] >= 0.99
    assert metrics["recall"] >= 0.99
    assert metrics["accuracy"] >= 0.99

    # Inferred class temperature should be compact and thermometer estimate should be close.
    assert np.all(class_frame["true_temp_span(K)"].to_numpy(dtype=np.float64) <= 0.8 + 1e-12)
    assert np.mean(class_frame["abs_estimation_error(K)"].to_numpy(dtype=np.float64)) < 0.08

    print(f"\nTransitivity triples checked: {triples_checked}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
