"""Minimal runnable MVP for Gravitational Wave Detectors.

Pipeline (compact and source-traceable):
1) Build a simplified binary-inspiral chirp template.
2) Simulate two interferometer streams (H1/L1) with colored noise.
3) Band-pass + PSD estimation (Welch), then matched filtering.
4) Extract coincidence features from two detectors.
5) Compare a rule-based trigger with sklearn and PyTorch classifiers.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GWConfig:
    """Configuration for a compact two-detector gravitational-wave MVP."""

    sample_rate_hz: int = 512
    duration_s: float = 2.0
    f_low_hz: float = 25.0
    f_high_hz: float = 230.0
    n_events: int = 240
    signal_probability: float = 0.5
    max_delay_ms: float = 8.0
    random_seed: int = 2026


def make_inspiral_template(cfg: GWConfig) -> np.ndarray:
    """Construct a toy inspiral chirp with rising frequency and amplitude."""

    n = int(cfg.sample_rate_hz * cfg.duration_s)
    t = np.linspace(0.0, cfg.duration_s, n, endpoint=False)
    tau = np.clip(cfg.duration_s - t, 1e-3, None)

    # Toy inspiral envelope: strain amplitude rises near merger.
    envelope = (tau / tau.max()) ** (-0.25)
    envelope = envelope / np.max(envelope)
    chirp = signal.chirp(t, f0=32.0, f1=220.0, t1=cfg.duration_s, method="quadratic", phi=90.0)
    template = envelope * chirp * signal.windows.tukey(n, alpha=0.15)

    # Normalize for easier SNR and amplitude control.
    return template / np.std(template)


def detector_noise_psd(freq_hz: np.ndarray) -> np.ndarray:
    """Simple analytic PSD shape to mimic colored interferometer noise."""

    f = np.maximum(freq_hz, 5.0)
    low_freq_wall = (40.0 / f) ** 4
    mid_band_floor = 1.0
    high_freq_rise = (f / 260.0) ** 2
    return low_freq_wall + mid_band_floor + high_freq_rise


def generate_colored_noise(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Generate colored Gaussian noise in frequency domain and transform back."""

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = detector_noise_psd(freqs)

    re = rng.normal(0.0, 1.0, size=freqs.size)
    im = rng.normal(0.0, 1.0, size=freqs.size)
    spectrum = (re + 1j * im) * np.sqrt(psd)
    spectrum[0] = rng.normal(0.0, 1.0) * math.sqrt(psd[0])
    if n % 2 == 0:
        spectrum[-1] = rng.normal(0.0, 1.0) * math.sqrt(psd[-1])

    noise = np.fft.irfft(spectrum, n=n)
    return noise / np.std(noise)


def time_shift_no_wrap(x: np.ndarray, shift_samples: int) -> np.ndarray:
    """Shift signal without circular wrap-around."""

    if shift_samples == 0:
        return x.copy()
    out = np.zeros_like(x)
    if shift_samples > 0:
        out[shift_samples:] = x[:-shift_samples]
    else:
        out[:shift_samples] = x[-shift_samples:]
    return out


def bandpass(data: np.ndarray, cfg: GWConfig) -> np.ndarray:
    """Band-pass filter in detector sensitive band."""

    sos = signal.butter(
        N=4,
        Wn=[cfg.f_low_hz, cfg.f_high_hz],
        btype="bandpass",
        output="sos",
        fs=cfg.sample_rate_hz,
    )
    return signal.sosfiltfilt(sos, data)


def estimate_psd(data: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one-sided PSD using Welch method."""

    freq, psd = signal.welch(
        data,
        fs=fs,
        window="hann",
        nperseg=256,
        noverlap=128,
        detrend=False,
        scaling="density",
    )
    return freq, np.clip(psd, 1e-6, None)


def matched_filter_snr(
    data: np.ndarray,
    template: np.ndarray,
    psd_freq: np.ndarray,
    psd_vals: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Compute matched-filter SNR time series in frequency domain."""

    n = len(data)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    df = freq[1] - freq[0]

    psd_interp = np.interp(freq, psd_freq, psd_vals, left=psd_vals[0], right=psd_vals[-1])
    psd_interp = np.clip(psd_interp, 1e-6, None)

    d_f = np.fft.rfft(data)
    h_f = np.fft.rfft(template)

    optimal = np.fft.irfft(d_f * np.conjugate(h_f) / psd_interp, n=n)
    norm_sq = 4.0 * np.sum((np.abs(h_f) ** 2) / psd_interp) * df
    norm = math.sqrt(max(norm_sq, 1e-12))
    snr = 4.0 * optimal * df / norm
    return snr


def inject_signal(
    template: np.ndarray,
    amplitude: float,
    shift_samples: int,
) -> np.ndarray:
    """Scaled and shifted chirp injection."""

    return amplitude * time_shift_no_wrap(template, shift_samples)


def build_event(
    template: np.ndarray,
    cfg: GWConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Simulate one H1/L1 event with or without a coherent injection."""

    n = template.size
    h1 = generate_colored_noise(n, cfg.sample_rate_hz, rng)
    l1 = generate_colored_noise(n, cfg.sample_rate_hz, rng)

    has_signal = int(rng.random() < cfg.signal_probability)
    true_delay_samples = 0
    if has_signal == 1:
        amp_base = rng.uniform(0.015, 0.090)
        amp_h1 = amp_base * rng.uniform(0.9, 1.1)
        amp_l1 = amp_base * rng.uniform(0.9, 1.1)
        max_delay_samples = int(round(cfg.max_delay_ms * cfg.sample_rate_hz / 1000.0))
        true_delay_samples = int(rng.integers(-max_delay_samples, max_delay_samples + 1))

        h1 = h1 + inject_signal(template, amp_h1, shift_samples=0)
        l1 = l1 + inject_signal(template, amp_l1, shift_samples=true_delay_samples)

    return h1, l1, has_signal, true_delay_samples


def feature_from_streams(
    h1: np.ndarray,
    l1: np.ndarray,
    template: np.ndarray,
    cfg: GWConfig,
) -> dict[str, float]:
    """Extract detector coincidence and matched-filter features."""

    h1_bp = bandpass(h1, cfg)
    l1_bp = bandpass(l1, cfg)
    tpl_bp = bandpass(template, cfg)

    f1, psd1 = estimate_psd(h1_bp, cfg.sample_rate_hz)
    f2, psd2 = estimate_psd(l1_bp, cfg.sample_rate_hz)

    snr1_ts = matched_filter_snr(h1_bp, tpl_bp, f1, psd1, cfg.sample_rate_hz)
    snr2_ts = matched_filter_snr(l1_bp, tpl_bp, f2, psd2, cfg.sample_rate_hz)

    idx1 = int(np.argmax(np.abs(snr1_ts)))
    idx2 = int(np.argmax(np.abs(snr2_ts)))
    snr1 = float(np.abs(snr1_ts[idx1]))
    snr2 = float(np.abs(snr2_ts[idx2]))

    raw_lag = abs(idx1 - idx2)
    lag_samples = min(raw_lag, len(snr1_ts) - raw_lag)
    lag_ms = 1000.0 * lag_samples / cfg.sample_rate_hz
    network_snr = float(math.sqrt(snr1**2 + snr2**2))

    corr = float(np.corrcoef(h1_bp, l1_bp)[0, 1])
    corr = corr if np.isfinite(corr) else 0.0
    snr_balance = float(min(snr1, snr2) / max(snr1, snr2, 1e-9))

    return {
        "snr_h1": snr1,
        "snr_l1": snr2,
        "network_snr": network_snr,
        "lag_ms": lag_ms,
        "corr_h1_l1": corr,
        "snr_balance": snr_balance,
    }


def build_dataset(cfg: GWConfig) -> pd.DataFrame:
    """Generate synthetic event table for detection experiments."""

    rng = np.random.default_rng(cfg.random_seed)
    template = make_inspiral_template(cfg)

    rows: list[dict[str, float | int]] = []
    for event_id in range(cfg.n_events):
        h1, l1, label, true_delay_samples = build_event(template, cfg, rng)
        features = feature_from_streams(h1, l1, template, cfg)
        row: dict[str, float | int] = {
            "event_id": event_id,
            "label": label,
            "true_delay_ms": 1000.0 * true_delay_samples / cfg.sample_rate_hz,
        }
        row.update(features)
        rows.append(row)
    return pd.DataFrame(rows)


def rule_based_detect(df: pd.DataFrame) -> np.ndarray:
    """Classic trigger: dual-detector SNR threshold + coincidence window."""

    return (
        (df["snr_h1"].to_numpy() >= 5.0)
        & (df["snr_l1"].to_numpy() >= 5.0)
        & (df["lag_ms"].to_numpy() <= 10.0)
    ).astype(int)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    """Return standard binary metrics used in detection diagnostics."""

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, score)
    else:
        out["roc_auc"] = float("nan")
    return out


def sklearn_detector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train logistic detector and return probabilities + hard predictions."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return prob, pred


class TinyMLP(torch.nn.Module):
    """Small PyTorch classifier for detector features."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def torch_detector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a tiny MLP with BCE loss and return probabilities + labels."""

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    x_train_t = torch.tensor(x_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    x_test_t = torch.tensor(x_test_s, dtype=torch.float32)

    torch.manual_seed(42)
    model = TinyMLP(in_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(220):
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = model(x_test_t).squeeze(1)
        prob = torch.sigmoid(test_logits).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def main() -> None:
    cfg = GWConfig()
    df = build_dataset(cfg)

    feature_cols = ["snr_h1", "snr_l1", "network_snr", "lag_ms", "corr_h1_l1", "snr_balance"]
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        np.arange(len(df)),
        test_size=0.30,
        random_state=7,
        stratify=y,
    )

    rule_pred_all = rule_based_detect(df)
    rule_metrics = evaluate_binary(y, rule_pred_all, score=rule_pred_all.astype(float))

    sk_prob, sk_pred = sklearn_detector(x_train, y_train, x_test)
    sk_metrics = evaluate_binary(y_test, sk_pred, score=sk_prob)

    torch_prob, torch_pred = torch_detector(x_train, y_train, x_test)
    torch_metrics = evaluate_binary(y_test, torch_pred, score=torch_prob)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")

    print("=== Gravitational Wave Detectors MVP (PHYS-0373) ===")
    print(
        f"Events={cfg.n_events}, fs={cfg.sample_rate_hz} Hz, duration={cfg.duration_s:.1f} s, "
        f"band=[{cfg.f_low_hz:.0f}, {cfg.f_high_hz:.0f}] Hz"
    )
    print()
    print("Sample events (first 8):")
    print(df.loc[:7, ["event_id", "label", "true_delay_ms", *feature_cols]])
    print()

    print("Rule-based trigger metrics (all events):")
    print(pd.Series(rule_metrics))
    print()

    print("Sklearn logistic metrics (test split):")
    print(pd.Series(sk_metrics))
    print()

    print("PyTorch tiny-MLP metrics (test split):")
    print(pd.Series(torch_metrics))
    print()

    test_table = df.iloc[idx_test].copy()
    test_table["sk_prob"] = sk_prob
    test_table["torch_prob"] = torch_prob
    top_candidates = test_table.sort_values("network_snr", ascending=False).head(6)
    print("Top network-SNR test candidates:")
    print(
        top_candidates[
            ["event_id", "label", "network_snr", "snr_h1", "snr_l1", "lag_ms", "sk_prob", "torch_prob"]
        ]
    )


if __name__ == "__main__":
    main()
