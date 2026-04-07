"""Minimal runnable MVP for Bubble Chamber track reconstruction.

Pipeline:
1) synthesize bubble-chamber-like circular tracks in a uniform magnetic field,
2) fit circle curvature from noisy points,
3) infer charge sign from rotation direction,
4) estimate transverse momentum by pT = 0.3 * |q| * B * r,
5) perform simple particle ID from (pT, bubble density).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import optimize
except Exception:  # pragma: no cover - fallback for minimal environments
    optimize = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal environments
    SKLEARN_AVAILABLE = False


B_FIELD_T = 1.5
RNG_SEED = 435


@dataclass(frozen=True)
class ParticleSpec:
    name: str
    mass_gev: float
    charge_e: int


@dataclass
class TrackSample:
    track_id: int
    particle: str
    q_true: int
    p_true_gev: float
    bubble_density_obs: float
    x: np.ndarray
    y: np.ndarray


PARTICLE_LIBRARY: dict[str, ParticleSpec] = {
    "electron": ParticleSpec("electron", mass_gev=0.000511, charge_e=-1),
    "pion": ParticleSpec("pion", mass_gev=0.13957, charge_e=+1),
    "proton": ParticleSpec("proton", mass_gev=0.93827, charge_e=+1),
}


class NumpyKNN:
    """Small KNN fallback when scikit-learn is unavailable."""

    def __init__(self, n_neighbors: int = 9) -> None:
        self.n_neighbors = n_neighbors
        self._x_train = np.empty((0, 2))
        self._y_train = np.empty((0,), dtype=object)
        self._mean = np.zeros(2)
        self._std = np.ones(2)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyKNN":
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0)
        self._std[self._std < 1e-12] = 1.0
        self._x_train = (x - self._mean) / self._std
        self._y_train = y
        return self

    def _predict_one(self, x_row: np.ndarray) -> str:
        xz = (x_row - self._mean) / self._std
        distances = np.sqrt(np.sum((self._x_train - xz) ** 2, axis=1))
        idx = np.argpartition(distances, self.n_neighbors)[: self.n_neighbors]
        d = distances[idx]
        w = 1.0 / np.clip(d, 1e-8, None)
        labels = self._y_train[idx]
        score: dict[str, float] = {}
        for label, weight in zip(labels, w):
            key = str(label)
            score[key] = score.get(key, 0.0) + float(weight)
        return max(score.items(), key=lambda kv: kv[1])[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = [self._predict_one(row) for row in x]
        return np.asarray(pred, dtype=object)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean(pred == y))


def beta_from_momentum(p_gev: float, mass_gev: float) -> float:
    energy_gev = math.sqrt(p_gev * p_gev + mass_gev * mass_gev)
    return p_gev / energy_gev


def bubble_density_model(beta: float) -> float:
    """Simplified dE/dx proxy: bubble density ~ constant + 1/beta^2."""

    beta_eff = max(beta, 1e-3)
    return 40.0 + 55.0 / (beta_eff * beta_eff)


def simulate_track(
    track_id: int,
    spec: ParticleSpec,
    p_true_gev: float,
    rng: np.random.Generator,
    n_points: int = 56,
    point_noise_std_m: float = 0.0025,
) -> TrackSample:
    r_true_m = p_true_gev / (0.3 * abs(spec.charge_e) * abs(B_FIELD_T))
    xc, yc = rng.uniform(-0.25, 0.25, size=2)
    phi0 = rng.uniform(-math.pi, math.pi)
    arc_span = rng.uniform(0.85, 1.35)

    # For Bz>0: positive charge rotates clockwise (decreasing polar angle).
    orientation = -1 if spec.charge_e * B_FIELD_T > 0 else 1
    t = np.linspace(0.0, 1.0, n_points)
    theta = phi0 + orientation * arc_span * t

    x = xc + r_true_m * np.cos(theta) + rng.normal(0.0, point_noise_std_m, size=n_points)
    y = yc + r_true_m * np.sin(theta) + rng.normal(0.0, point_noise_std_m, size=n_points)

    beta = beta_from_momentum(p_true_gev, spec.mass_gev)
    bubble_density_true = bubble_density_model(beta)
    bubble_density_obs = float(max(1.0, bubble_density_true + rng.normal(0.0, 6.0)))

    return TrackSample(
        track_id=track_id,
        particle=spec.name,
        q_true=spec.charge_e,
        p_true_gev=float(p_true_gev),
        bubble_density_obs=bubble_density_obs,
        x=x,
        y=y,
    )


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a circle from noisy points: algebraic init + optional nonlinear refine."""

    if x.size < 6:
        raise ValueError("need at least 6 points for stable circle fitting")

    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    xc0, yc0, c0 = params
    r0 = math.sqrt(max(xc0 * xc0 + yc0 * yc0 + c0, 1e-12))

    if optimize is None:
        residual = np.sqrt((x - xc0) ** 2 + (y - yc0) ** 2) - r0
        rms = float(np.sqrt(np.mean(residual * residual)))
        return float(xc0), float(yc0), float(abs(r0)), rms

    def residual_fun(v: np.ndarray) -> np.ndarray:
        xc, yc, r = v
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    res = optimize.least_squares(
        residual_fun,
        x0=np.asarray([xc0, yc0, r0], dtype=float),
        method="trf",
    )
    xc, yc, r = res.x
    rms = float(np.sqrt(np.mean(res.fun * res.fun)))
    return float(xc), float(yc), float(abs(r)), rms


def infer_charge_sign(x: np.ndarray, y: np.ndarray, xc: float, yc: float) -> int:
    """Infer charge sign from track rotation direction around fitted center."""

    theta = np.unwrap(np.arctan2(y - yc, x - xc))
    dtheta = np.diff(theta)
    if dtheta.size == 0:
        return 0

    mean_dtheta = float(np.mean(dtheta))
    if abs(mean_dtheta) < 1e-10:
        return 0

    orientation = 1 if mean_dtheta > 0.0 else -1  # +1 = CCW, -1 = CW
    b_sign = 1 if B_FIELD_T > 0.0 else -1
    q_est = -orientation * b_sign
    return int(q_est)


def build_pid_classifier(
    rng: np.random.Generator,
    n_samples: int = 3200,
) -> tuple[Any, float]:
    names = list(PARTICLE_LIBRARY.keys())
    feats: list[tuple[float, float]] = []
    labels: list[str] = []

    for _ in range(n_samples):
        name = str(rng.choice(names))
        spec = PARTICLE_LIBRARY[name]
        p_true = float(rng.uniform(0.15, 1.50))

        beta = beta_from_momentum(p_true, spec.mass_gev)
        bubble_density = bubble_density_model(beta) + rng.normal(0.0, 8.0)
        p_meas = max(0.05, p_true * (1.0 + rng.normal(0.0, 0.06)))

        feats.append((p_meas, bubble_density))
        labels.append(name)

    x = np.asarray(feats, dtype=float)
    y = np.asarray(labels, dtype=object)

    if SKLEARN_AVAILABLE:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=RNG_SEED,
            stratify=y,
        )
        clf = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=11, weights="distance"),
        )
        clf.fit(x_train, y_train)
        val_acc = float(clf.score(x_test, y_test))
        return clf, val_acc

    split = int(0.75 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    clf = NumpyKNN(n_neighbors=11).fit(x_train, y_train)
    val_acc = clf.score(x_test, y_test)
    return clf, float(val_acc)


def reconstruct_tracks(tracks: list[TrackSample], clf: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for tr in tracks:
        xc, yc, r_est, fit_rms = fit_circle(tr.x, tr.y)
        q_est = infer_charge_sign(tr.x, tr.y, xc, yc)
        p_est = 0.3 * abs(B_FIELD_T) * r_est

        pid_pred = str(clf.predict(np.asarray([[p_est, tr.bubble_density_obs]], dtype=float))[0])
        r_true = tr.p_true_gev / (0.3 * abs(B_FIELD_T))

        rows.append(
            {
                "track_id": tr.track_id,
                "pid_true": tr.particle,
                "pid_pred": pid_pred,
                "q_true": tr.q_true,
                "q_est": q_est,
                "p_true_gev": tr.p_true_gev,
                "p_est_gev": p_est,
                "r_true_m": r_true,
                "r_est_m": r_est,
                "bubble_density_obs": tr.bubble_density_obs,
                "fit_rms_m": fit_rms,
                "rel_p_error": (p_est - tr.p_true_gev) / tr.p_true_gev,
            }
        )

    return pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)


def generate_event_tracks(rng: np.random.Generator) -> list[TrackSample]:
    plan = [
        ("electron", (0.20, 0.55)),
        ("pion", (0.30, 0.90)),
        ("proton", (0.45, 1.25)),
        ("pion", (0.22, 0.75)),
        ("electron", (0.25, 0.60)),
        ("proton", (0.40, 1.10)),
    ]

    tracks: list[TrackSample] = []
    for i, (name, (p_low, p_high)) in enumerate(plan, start=1):
        spec = PARTICLE_LIBRARY[name]
        p_true = float(rng.uniform(p_low, p_high))
        tracks.append(simulate_track(i, spec, p_true, rng))
    return tracks


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    classifier, val_acc = build_pid_classifier(rng)
    tracks = generate_event_tracks(rng)
    result_df = reconstruct_tracks(tracks, classifier)

    charge_acc = float(np.mean(result_df["q_true"].to_numpy() == result_df["q_est"].to_numpy()))
    pid_acc = float(np.mean(result_df["pid_true"].to_numpy() == result_df["pid_pred"].to_numpy()))
    mean_abs_rel_p = float(np.mean(np.abs(result_df["rel_p_error"].to_numpy())))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")

    print("Bubble Chamber MVP: curvature-based momentum + simple PID")
    print(f"Magnetic field B = {B_FIELD_T:.2f} T")
    print(f"PID classifier validation accuracy = {val_acc:.3f}")

    show_cols = [
        "track_id",
        "pid_true",
        "pid_pred",
        "q_true",
        "q_est",
        "p_true_gev",
        "p_est_gev",
        "r_true_m",
        "r_est_m",
        "bubble_density_obs",
        "fit_rms_m",
        "rel_p_error",
    ]
    print(result_df[show_cols])
    print(
        "Event summary: "
        f"charge_acc={charge_acc:.3f}, pid_acc={pid_acc:.3f}, mean_abs_rel_p_error={mean_abs_rel_p:.3f}"
    )


if __name__ == "__main__":
    main()
