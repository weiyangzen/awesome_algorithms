"""Minimal runnable MVP for b-tagging in particle physics.

This script builds a toy jet dataset with b / c / light flavor labels,
trains a logistic b-tagger implemented directly in NumPy,
and reports ROC-AUC plus mistag rates at fixed b-efficiency working points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_NAMES: Tuple[str, ...] = (
    "ip3d_sig",
    "sv_mass",
    "sv_flight_sig",
    "n_sv_tracks",
    "soft_lepton_ptrel",
    "jet_pt",
    "jet_width",
)


@dataclass(frozen=True)
class LogisticModel:
    """Simple logistic regression model parameters."""

    weights: np.ndarray
    bias: float
    train_loss: Tuple[float, ...]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature mean/std for z-score normalization."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std_safe = np.where(std < 1.0e-9, 1.0, std)
    return mean, std_safe


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalization."""
    return (x - mean) / std


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average binary cross entropy."""
    eps = 1.0e-9
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def fit_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    lr: float = 0.08,
    epochs: int = 1200,
    l2: float = 1.0e-3,
) -> LogisticModel:
    """Train logistic regression with full-batch gradient descent."""
    n_samples, n_features = x_train.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    checkpoints: List[float] = []
    checkpoint_stride = max(epochs // 10, 1)

    for step in range(epochs):
        logits = x_train @ w + b
        prob = sigmoid(logits)

        error = prob - y_train
        grad_w = (x_train.T @ error) / n_samples + l2 * w
        grad_b = float(np.mean(error))

        w -= lr * grad_w
        b -= lr * grad_b

        if step % checkpoint_stride == 0 or step == epochs - 1:
            loss = binary_cross_entropy(y_train, sigmoid(x_train @ w + b))
            checkpoints.append(loss)

    return LogisticModel(weights=w, bias=b, train_loss=tuple(checkpoints))


def predict_proba(model: LogisticModel, x: np.ndarray) -> np.ndarray:
    """Predict P(b-jet | features)."""
    return sigmoid(x @ model.weights + model.bias)


def generate_toy_btag_dataframe(seed: int = 426, n_b: int = 2200, n_c: int = 1400, n_light: int = 2400) -> pd.DataFrame:
    """Generate a toy jet sample with physically motivated feature shapes.

    Features:
    - ip3d_sig: signed 3D impact parameter significance
    - sv_mass: reconstructed secondary-vertex invariant mass (GeV)
    - sv_flight_sig: secondary-vertex flight distance significance
    - n_sv_tracks: number of tracks attached to SV
    - soft_lepton_ptrel: soft-lepton pT relative to jet axis (GeV)
    - jet_pt: jet transverse momentum (GeV)
    - jet_width: radial energy spread in (eta, phi)
    """
    rng = np.random.default_rng(seed)

    def make_block(flavor: str, n: int) -> pd.DataFrame:
        if flavor == "b":
            ip3d_sig = rng.normal(loc=3.8, scale=1.3, size=n)
            sv_mass = rng.normal(loc=1.9, scale=0.60, size=n)
            sv_flight_sig = rng.normal(loc=7.5, scale=2.1, size=n)
            n_sv_tracks = rng.poisson(lam=4.5, size=n)
            soft_lepton_ptrel = rng.gamma(shape=1.9, scale=1.3, size=n)
            jet_pt = rng.lognormal(mean=4.28, sigma=0.45, size=n)
            jet_width = rng.normal(loc=0.102, scale=0.022, size=n)
            is_b = np.ones(n, dtype=int)
        elif flavor == "c":
            ip3d_sig = rng.normal(loc=2.7, scale=1.2, size=n)
            sv_mass = rng.normal(loc=1.3, scale=0.45, size=n)
            sv_flight_sig = rng.normal(loc=4.8, scale=1.8, size=n)
            n_sv_tracks = rng.poisson(lam=3.4, size=n)
            soft_lepton_ptrel = rng.gamma(shape=1.6, scale=1.2, size=n)
            jet_pt = rng.lognormal(mean=4.10, sigma=0.45, size=n)
            jet_width = rng.normal(loc=0.116, scale=0.024, size=n)
            is_b = np.zeros(n, dtype=int)
        elif flavor == "light":
            ip3d_sig = rng.normal(loc=1.4, scale=1.0, size=n)
            sv_mass = rng.normal(loc=0.55, scale=0.28, size=n)
            sv_flight_sig = rng.normal(loc=2.4, scale=1.1, size=n)
            n_sv_tracks = rng.poisson(lam=1.8, size=n)
            soft_lepton_ptrel = rng.gamma(shape=1.2, scale=1.0, size=n)
            jet_pt = rng.lognormal(mean=4.00, sigma=0.52, size=n)
            jet_width = rng.normal(loc=0.135, scale=0.028, size=n)
            is_b = np.zeros(n, dtype=int)
        else:
            raise ValueError(f"Unknown flavor: {flavor}")

        # Add weak correlations to make the toy problem less trivial.
        ip3d_sig = ip3d_sig + 0.03 * np.log1p(jet_pt) + rng.normal(0.0, 0.35, size=n)
        sv_flight_sig = (
            sv_flight_sig
            + 0.08 * np.sqrt(np.maximum(sv_mass, 0.0))
            + rng.normal(0.0, 0.45, size=n)
        )
        soft_lepton_ptrel = (
            soft_lepton_ptrel
            + 0.006 * np.maximum(jet_pt - 30.0, 0.0)
            + rng.normal(0.0, 0.30, size=n)
        )

        frame = pd.DataFrame(
            {
                "flavor": flavor,
                "is_b": is_b,
                "ip3d_sig": np.clip(ip3d_sig, 0.0, None),
                "sv_mass": np.clip(sv_mass, 0.0, None),
                "sv_flight_sig": np.clip(sv_flight_sig, 0.0, None),
                "n_sv_tracks": np.clip(n_sv_tracks.astype(float), 0.0, None),
                "soft_lepton_ptrel": np.clip(soft_lepton_ptrel, 0.0, None),
                "jet_pt": np.clip(jet_pt, 5.0, None),
                "jet_width": np.clip(jet_width, 0.02, None),
            }
        )
        return frame

    df = pd.concat(
        [
            make_block("b", n_b),
            make_block("c", n_c),
            make_block("light", n_light),
        ],
        ignore_index=True,
    )

    # Deterministic row permutation for train/test split stability.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def threshold_for_target_efficiency(y_true: np.ndarray, y_score: np.ndarray, target_eff: float) -> float:
    """Choose threshold giving approximately target b-jet efficiency."""
    if not (0.0 < target_eff <= 1.0):
        raise ValueError("target_eff must be in (0, 1].")

    pos_scores = y_score[y_true == 1]
    if len(pos_scores) == 0:
        raise ValueError("No positive samples in y_true.")

    quantile = 1.0 - target_eff
    return float(np.quantile(pos_scores, quantile))


def evaluate_working_point(
    y_true: np.ndarray,
    y_score: np.ndarray,
    flavors: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Return b-efficiency and mistag rates at a threshold."""
    select = y_score >= threshold

    mask_b = y_true == 1
    mask_nonb = y_true == 0
    mask_c = flavors == "c"
    mask_light = flavors == "light"

    b_eff = float(np.mean(select[mask_b])) if np.any(mask_b) else float("nan")
    mistag_nonb = float(np.mean(select[mask_nonb])) if np.any(mask_nonb) else float("nan")
    mistag_c = float(np.mean(select[mask_c])) if np.any(mask_c) else float("nan")
    mistag_light = float(np.mean(select[mask_light])) if np.any(mask_light) else float("nan")

    return {
        "threshold": threshold,
        "b_eff": b_eff,
        "mistag_nonb": mistag_nonb,
        "mistag_c": mistag_c,
        "mistag_light": mistag_light,
    }


def coefficient_table(model: LogisticModel, feature_names: Sequence[str]) -> pd.DataFrame:
    """Tabulate learned coefficients by absolute magnitude."""
    table = pd.DataFrame(
        {
            "feature": list(feature_names),
            "weight": model.weights,
            "abs_weight": np.abs(model.weights),
        }
    )
    return table.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def main() -> None:
    """Run deterministic toy b-tagging demo end-to-end."""
    df = generate_toy_btag_dataframe(seed=426, n_b=2200, n_c=1400, n_light=2400)

    x_all = df.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    y_all = df["is_b"].to_numpy(dtype=int)
    flavor_all = df["flavor"].to_numpy(dtype=str)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=426,
        stratify=y_all,
    )

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]
    flavor_test = flavor_all[test_idx]

    mean, std = standardize_fit(x_train)
    x_train_z = standardize_apply(x_train, mean, std)
    x_test_z = standardize_apply(x_test, mean, std)

    model = fit_logistic_regression(
        x_train=x_train_z,
        y_train=y_train,
        lr=0.08,
        epochs=1200,
        l2=1.0e-3,
    )

    test_score = predict_proba(model, x_test_z)
    auc = roc_auc_score(y_test, test_score)

    wp70_thr = threshold_for_target_efficiency(y_test, test_score, target_eff=0.70)
    wp80_thr = threshold_for_target_efficiency(y_test, test_score, target_eff=0.80)

    wp70 = evaluate_working_point(y_test, test_score, flavor_test, threshold=wp70_thr)
    wp80 = evaluate_working_point(y_test, test_score, flavor_test, threshold=wp80_thr)

    wp_table = pd.DataFrame(
        [
            {"working_point": "70% b-eff", **wp70},
            {"working_point": "80% b-eff", **wp80},
        ]
    )

    coeff_df = coefficient_table(model, FEATURE_NAMES)
    class_counts = df.groupby("flavor", as_index=False).size().rename(columns={"size": "n_jets"})

    print("=== Toy b-tagging dataset summary ===")
    print(class_counts.to_string(index=False))
    print()

    print("=== Model quality ===")
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    print(f"ROC-AUC (test): {auc:.4f}")
    print(f"Training loss checkpoints: {[round(x, 4) for x in model.train_loss]}")
    print()

    print("=== Working points ===")
    print(
        wp_table[["working_point", "threshold", "b_eff", "mistag_nonb", "mistag_c", "mistag_light"]].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )
    print()

    print("=== Learned coefficient ranking (|weight| descending) ===")
    print(coeff_df[["feature", "weight"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
