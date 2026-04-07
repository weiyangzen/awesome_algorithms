"""Minimal runnable MVP for the Quark Model.

This script builds a transparent constituent-quark mass model:
1) encode hadrons by valence quarks and total spin,
2) fit channel-dependent offsets and hyperfine couplings,
3) predict hadron masses,
4) verify spin-ordering and quantum-number consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuarkFlavor:
    mass_gev: float
    charge_e: float


@dataclass(frozen=True)
class HadronSample:
    name: str
    kind: str  # "meson" or "baryon"
    constituents: tuple[str, ...]
    total_spin: float
    observed_mass_gev: float
    split: str  # "train" or "test"


@dataclass(frozen=True)
class ModelParams:
    delta_meson: float
    delta_baryon: float
    a_meson: float
    a_baryon: float


QUARK_DB: dict[str, QuarkFlavor] = {
    "u": QuarkFlavor(mass_gev=0.336, charge_e=2.0 / 3.0),
    "d": QuarkFlavor(mass_gev=0.340, charge_e=-1.0 / 3.0),
    "s": QuarkFlavor(mass_gev=0.486, charge_e=-1.0 / 3.0),
    "c": QuarkFlavor(mass_gev=1.550, charge_e=2.0 / 3.0),
    "b": QuarkFlavor(mass_gev=4.850, charge_e=-1.0 / 3.0),
}


DATASET: tuple[HadronSample, ...] = (
    HadronSample("pi+", "meson", ("u", "d_bar"), 0.0, 0.1396, "train"),
    HadronSample("rho+", "meson", ("u", "d_bar"), 1.0, 0.7753, "train"),
    HadronSample("K+", "meson", ("u", "s_bar"), 0.0, 0.4937, "train"),
    HadronSample("K*+", "meson", ("u", "s_bar"), 1.0, 0.8917, "train"),
    HadronSample("D+", "meson", ("c", "d_bar"), 0.0, 1.8697, "train"),
    HadronSample("D*+", "meson", ("c", "d_bar"), 1.0, 2.0103, "train"),
    HadronSample("B+", "meson", ("u", "b_bar"), 0.0, 5.2793, "train"),
    HadronSample("B*+", "meson", ("u", "b_bar"), 1.0, 5.3247, "test"),
    HadronSample("eta_c", "meson", ("c", "c_bar"), 0.0, 2.9839, "train"),
    HadronSample("J/psi", "meson", ("c", "c_bar"), 1.0, 3.0969, "test"),
    HadronSample("proton", "baryon", ("u", "u", "d"), 0.5, 0.9383, "train"),
    HadronSample("neutron", "baryon", ("u", "d", "d"), 0.5, 0.9396, "train"),
    HadronSample("Lambda", "baryon", ("u", "d", "s"), 0.5, 1.1157, "train"),
    HadronSample("Sigma+", "baryon", ("u", "u", "s"), 0.5, 1.1894, "train"),
    HadronSample("Sigma*+", "baryon", ("u", "u", "s"), 1.5, 1.3828, "test"),
    HadronSample("Delta++", "baryon", ("u", "u", "u"), 1.5, 1.2320, "train"),
    HadronSample("Omega-", "baryon", ("s", "s", "s"), 1.5, 1.6720, "test"),
    HadronSample("Lambda_c+", "baryon", ("u", "d", "c"), 0.5, 2.2865, "test"),
)


def is_antiquark(label: str) -> bool:
    return label.endswith("_bar")


def flavor_name(label: str) -> str:
    return label[:-4] if is_antiquark(label) else label


def constituent_mass_gev(label: str) -> float:
    flavor = flavor_name(label)
    if flavor not in QUARK_DB:
        raise KeyError(f"Unknown quark flavor: {label}")
    return QUARK_DB[flavor].mass_gev


def constituent_charge_e(label: str) -> float:
    flavor = flavor_name(label)
    q = QUARK_DB[flavor].charge_e
    return -q if is_antiquark(label) else q


def constituent_baryon_number(label: str) -> float:
    return -1.0 / 3.0 if is_antiquark(label) else 1.0 / 3.0


def validate_sample(sample: HadronSample) -> None:
    if sample.kind not in {"meson", "baryon"}:
        raise ValueError(f"Unknown hadron kind: {sample.kind}")
    if sample.kind == "meson":
        if len(sample.constituents) != 2:
            raise ValueError(f"Meson {sample.name} must have two constituents")
        anti_count = sum(int(is_antiquark(x)) for x in sample.constituents)
        if anti_count != 1:
            raise ValueError(f"Meson {sample.name} must have one quark and one antiquark")
        if sample.total_spin not in {0.0, 1.0}:
            raise ValueError(f"Meson {sample.name} spin must be 0 or 1")
    else:
        if len(sample.constituents) != 3:
            raise ValueError(f"Baryon {sample.name} must have three constituents")
        if any(is_antiquark(x) for x in sample.constituents):
            raise ValueError(f"Baryon {sample.name} should not contain antiquarks in this MVP")
        if sample.total_spin not in {0.5, 1.5}:
            raise ValueError(f"Baryon {sample.name} spin must be 1/2 or 3/2")


def base_mass_gev(sample: HadronSample) -> float:
    return float(sum(constituent_mass_gev(q) for q in sample.constituents))


def spin_coupling_factor(sample: HadronSample) -> float:
    if sample.kind == "meson":
        return -0.75 if sample.total_spin == 0.0 else 0.25
    return -0.75 if sample.total_spin == 0.5 else 0.75


def inverse_mass_pair_sum(sample: HadronSample) -> float:
    masses = [constituent_mass_gev(q) for q in sample.constituents]
    return float(sum(1.0 / (masses[i] * masses[j]) for i, j in combinations(range(len(masses)), 2)))


def hyperfine_feature(sample: HadronSample) -> float:
    return spin_coupling_factor(sample) * inverse_mass_pair_sum(sample)


def build_linear_system(samples: list[HadronSample]) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []

    for sample in samples:
        h = hyperfine_feature(sample)
        residual_target = sample.observed_mass_gev - base_mass_gev(sample)

        if sample.kind == "meson":
            x_rows.append([1.0, 0.0, h, 0.0])
        else:
            x_rows.append([0.0, 1.0, 0.0, h])
        y_rows.append(residual_target)

    return np.array(x_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64)


def fit_model(train_samples: list[HadronSample]) -> ModelParams:
    x, y = build_linear_system(train_samples)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return ModelParams(
        delta_meson=float(beta[0]),
        delta_baryon=float(beta[1]),
        a_meson=float(beta[2]),
        a_baryon=float(beta[3]),
    )


def predict_mass_gev(sample: HadronSample, params: ModelParams) -> float:
    base = base_mass_gev(sample)
    h = hyperfine_feature(sample)

    if sample.kind == "meson":
        return base + params.delta_meson + params.a_meson * h
    return base + params.delta_baryon + params.a_baryon * h


def hadron_quantum_numbers(sample: HadronSample) -> tuple[float, float]:
    charge = float(sum(constituent_charge_e(q) for q in sample.constituents))
    baryon_number = float(sum(constituent_baryon_number(q) for q in sample.constituents))
    return charge, baryon_number


def evaluate_samples(samples: list[HadronSample], params: ModelParams) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for sample in samples:
        pred = predict_mass_gev(sample, params)
        charge, baryon_number = hadron_quantum_numbers(sample)
        error = pred - sample.observed_mass_gev

        rows.append(
            {
                "name": sample.name,
                "kind": sample.kind,
                "split": sample.split,
                "spin": sample.total_spin,
                "observed_mass_gev": sample.observed_mass_gev,
                "predicted_mass_gev": pred,
                "error_gev": error,
                "abs_error_gev": abs(error),
                "charge_e": charge,
                "baryon_number": baryon_number,
                "base_mass_gev": base_mass_gev(sample),
                "hyperfine_feature": hyperfine_feature(sample),
            }
        )

    return pd.DataFrame(rows)


def mean_abs_error(df: pd.DataFrame, split: str) -> float:
    mask = df["split"] == split
    return float(df.loc[mask, "abs_error_gev"].mean())


def check_spin_ordering(df: pd.DataFrame) -> dict[str, float]:
    lookup = {row["name"]: float(row["predicted_mass_gev"]) for _, row in df.iterrows()}
    pairs = [
        ("rho+", "pi+"),
        ("K*+", "K+"),
        ("D*+", "D+"),
        ("B*+", "B+"),
        ("J/psi", "eta_c"),
        ("Sigma*+", "Sigma+"),
    ]
    return {f"{hi}>{lo}": lookup[hi] - lookup[lo] for hi, lo in pairs}


def check_quantum_numbers(df: pd.DataFrame) -> tuple[float, float]:
    expected_charge = {
        "pi+": 1.0,
        "rho+": 1.0,
        "K+": 1.0,
        "K*+": 1.0,
        "D+": 1.0,
        "D*+": 1.0,
        "B+": 1.0,
        "B*+": 1.0,
        "eta_c": 0.0,
        "J/psi": 0.0,
        "proton": 1.0,
        "neutron": 0.0,
        "Lambda": 0.0,
        "Sigma+": 1.0,
        "Sigma*+": 1.0,
        "Delta++": 2.0,
        "Omega-": -1.0,
        "Lambda_c+": 1.0,
    }

    expected_baryon = {
        row["name"]: (0.0 if row["kind"] == "meson" else 1.0)
        for _, row in df.iterrows()
    }

    max_charge_dev = 0.0
    max_baryon_dev = 0.0
    for _, row in df.iterrows():
        name = str(row["name"])
        max_charge_dev = max(max_charge_dev, abs(float(row["charge_e"]) - expected_charge[name]))
        max_baryon_dev = max(max_baryon_dev, abs(float(row["baryon_number"]) - expected_baryon[name]))

    return float(max_charge_dev), float(max_baryon_dev)


def main() -> None:
    for sample in DATASET:
        validate_sample(sample)

    train_samples = [s for s in DATASET if s.split == "train"]
    test_samples = [s for s in DATASET if s.split == "test"]

    params = fit_model(train_samples)
    report_df = evaluate_samples(list(DATASET), params)

    train_mae = mean_abs_error(report_df, split="train")
    test_mae = mean_abs_error(report_df, split="test")
    spin_gaps = check_spin_ordering(report_df)
    max_charge_dev, max_baryon_dev = check_quantum_numbers(report_df)

    print("=== Fitted Quark-Model Parameters ===")
    print(f"delta_meson  = {params.delta_meson:.6f} GeV")
    print(f"delta_baryon = {params.delta_baryon:.6f} GeV")
    print(f"a_meson      = {params.a_meson:.6f} GeV^3")
    print(f"a_baryon     = {params.a_baryon:.6f} GeV^3")
    print()

    print("=== Hadron Mass Report ===")
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    print("=== Summary Metrics ===")
    print(f"train MAE = {train_mae:.6f} GeV")
    print(f"test  MAE = {test_mae:.6f} GeV")
    print(f"max charge deviation = {max_charge_dev:.3e}")
    print(f"max baryon-number deviation = {max_baryon_dev:.3e}")
    for label, gap in spin_gaps.items():
        print(f"spin splitting {label}: {gap:.6f} GeV")

    assert not report_df["predicted_mass_gev"].isna().any(), "Predicted masses contain NaN"
    assert params.a_meson > 0.0 and params.a_baryon > 0.0, "Hyperfine couplings should be positive"
    assert train_mae < 0.08, f"Train MAE too large: {train_mae:.6f}"
    assert test_mae < 0.13, f"Test MAE too large: {test_mae:.6f}"
    assert max_charge_dev < 1e-12, f"Charge consistency check failed: {max_charge_dev:.3e}"
    assert max_baryon_dev < 1e-12, f"Baryon number check failed: {max_baryon_dev:.3e}"
    assert all(gap > 0.0 for gap in spin_gaps.values()), "Spin-ordering check failed"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
