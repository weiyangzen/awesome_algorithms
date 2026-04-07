"""Minimal runnable MVP for Conditional Random Field (CRF), MATH-0324.

This script implements a first-order linear-chain CRF from scratch with NumPy:
- log-space forward/backward for partition function and marginals,
- gradient of conditional log-likelihood,
- gradient ascent training,
- Viterbi decoding for sequence prediction.

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class CRFParams:
    """Parameters of a first-order linear-chain CRF."""

    start: np.ndarray  # shape (K,)
    end: np.ndarray  # shape (K,)
    trans: np.ndarray  # shape (K, K), trans[i, j]=score(y_{t-1}=i -> y_t=j)
    emit: np.ndarray  # shape (K, V), emit[k, obs]=score(y_t=k, x_t=obs)


@dataclass
class ForwardBackwardCache:
    """Cached dynamic-programming tables for one sequence."""

    alpha: np.ndarray  # shape (T, K)
    beta: np.ndarray  # shape (T, K)
    log_z: float


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    m = np.max(values, axis=axis, keepdims=True)
    stable = values - m
    s = np.sum(np.exp(stable), axis=axis, keepdims=True)
    out = m + np.log(s + 1e-300)
    if axis is None:
        return np.asarray(out).reshape(())
    return np.squeeze(out, axis=axis)


def score_sequence(params: CRFParams, obs: np.ndarray, tags: np.ndarray) -> float:
    """Unnormalized path score s(x,y) for one labeled sequence."""
    t_len = obs.shape[0]
    score = float(params.start[tags[0]] + params.emit[tags[0], obs[0]])
    for t in range(1, t_len):
        score += float(params.trans[tags[t - 1], tags[t]] + params.emit[tags[t], obs[t]])
    score += float(params.end[tags[-1]])
    return score


def forward_log(params: CRFParams, obs: np.ndarray) -> Tuple[np.ndarray, float]:
    """Forward DP in log-space; returns alpha and log Z(x)."""
    t_len = obs.shape[0]
    n_tags = params.start.shape[0]

    alpha = np.zeros((t_len, n_tags), dtype=np.float64)
    alpha[0] = params.start + params.emit[:, obs[0]]

    for t in range(1, t_len):
        for j in range(n_tags):
            alpha[t, j] = params.emit[j, obs[t]] + float(
                logsumexp(alpha[t - 1] + params.trans[:, j])
            )

    log_z = float(logsumexp(alpha[-1] + params.end))
    return alpha, log_z


def backward_log(params: CRFParams, obs: np.ndarray) -> np.ndarray:
    """Backward DP in log-space; returns beta."""
    t_len = obs.shape[0]
    n_tags = params.start.shape[0]

    beta = np.zeros((t_len, n_tags), dtype=np.float64)
    beta[-1] = params.end

    for t in range(t_len - 2, -1, -1):
        for i in range(n_tags):
            beta[t, i] = float(
                logsumexp(params.trans[i] + params.emit[:, obs[t + 1]] + beta[t + 1])
            )
    return beta


def forward_backward(params: CRFParams, obs: np.ndarray) -> ForwardBackwardCache:
    """Compute alpha/beta/logZ for one observation sequence."""
    alpha, log_z = forward_log(params, obs)
    beta = backward_log(params, obs)
    return ForwardBackwardCache(alpha=alpha, beta=beta, log_z=log_z)


def marginals(
    params: CRFParams,
    obs: np.ndarray,
    fb: ForwardBackwardCache,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute node marginals gamma and edge marginals xi.

    Returns:
        gamma: shape (T, K), gamma[t,k]=P(y_t=k|x)
        xi: shape (T-1, K, K), xi[t,i,j]=P(y_t=i,y_{t+1}=j|x)
    """
    t_len, n_tags = fb.alpha.shape

    gamma_log = fb.alpha + fb.beta - fb.log_z
    gamma = np.exp(gamma_log)
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi = np.zeros((t_len - 1, n_tags, n_tags), dtype=np.float64)
    for t in range(t_len - 1):
        log_numer = (
            fb.alpha[t][:, None]
            + params.trans
            + params.emit[:, obs[t + 1]][None, :]
            + fb.beta[t + 1][None, :]
            - fb.log_z
        )
        xi_t = np.exp(log_numer)
        xi_t /= xi_t.sum()
        xi[t] = xi_t

    return gamma, xi


def viterbi_decode(params: CRFParams, obs: np.ndarray) -> np.ndarray:
    """Viterbi decoding for argmax_y s(x,y)."""
    t_len = obs.shape[0]
    n_tags = params.start.shape[0]

    delta = np.zeros((t_len, n_tags), dtype=np.float64)
    psi = np.zeros((t_len, n_tags), dtype=np.int64)

    delta[0] = params.start + params.emit[:, obs[0]]

    for t in range(1, t_len):
        for j in range(n_tags):
            candidates = delta[t - 1] + params.trans[:, j]
            best_prev = int(np.argmax(candidates))
            psi[t, j] = best_prev
            delta[t, j] = candidates[best_prev] + params.emit[j, obs[t]]

    best_last = int(np.argmax(delta[-1] + params.end))
    path = np.zeros(t_len, dtype=np.int64)
    path[-1] = best_last

    for t in range(t_len - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def sequence_gradients(
    params: CRFParams,
    obs: np.ndarray,
    gold: np.ndarray,
) -> Tuple[CRFParams, float, ForwardBackwardCache, np.ndarray, np.ndarray]:
    """Gradient of log p(y|x) for one sequence."""
    fb = forward_backward(params, obs)
    gamma, xi = marginals(params, obs, fb)

    grad = CRFParams(
        start=np.zeros_like(params.start),
        end=np.zeros_like(params.end),
        trans=np.zeros_like(params.trans),
        emit=np.zeros_like(params.emit),
    )

    # Empirical feature counts.
    grad.start[gold[0]] += 1.0
    grad.end[gold[-1]] += 1.0
    for t in range(obs.shape[0]):
        grad.emit[gold[t], obs[t]] += 1.0
    for t in range(obs.shape[0] - 1):
        grad.trans[gold[t], gold[t + 1]] += 1.0

    # Subtract expected feature counts under p(y|x).
    grad.start -= gamma[0]
    grad.end -= gamma[-1]

    for t in range(obs.shape[0]):
        grad.emit[:, obs[t]] -= gamma[t]
    for t in range(obs.shape[0] - 1):
        grad.trans -= xi[t]

    gold_score = score_sequence(params, obs, gold)
    log_likelihood = gold_score - fb.log_z

    return grad, float(log_likelihood), fb, gamma, xi


def add_scaled_params(dst: CRFParams, src: CRFParams, scale: float) -> None:
    """In-place dst += scale * src."""
    dst.start += scale * src.start
    dst.end += scale * src.end
    dst.trans += scale * src.trans
    dst.emit += scale * src.emit


def l2_apply_to_gradient(grad: CRFParams, params: CRFParams, l2: float) -> None:
    """For objective L - 0.5*l2*||theta||^2, grad <- grad - l2*theta."""
    if l2 <= 0.0:
        return
    grad.start -= l2 * params.start
    grad.end -= l2 * params.end
    grad.trans -= l2 * params.trans
    grad.emit -= l2 * params.emit


def train_crf(
    train_x: Sequence[np.ndarray],
    train_y: Sequence[np.ndarray],
    n_tags: int,
    n_obs: int,
    epochs: int = 80,
    lr: float = 0.35,
    l2: float = 5e-4,
    seed: int = 324,
) -> Tuple[CRFParams, List[float]]:
    """Train CRF by full-batch gradient ascent on conditional log-likelihood."""
    rng = np.random.default_rng(seed)

    params = CRFParams(
        start=rng.normal(0.0, 0.05, size=n_tags),
        end=rng.normal(0.0, 0.05, size=n_tags),
        trans=rng.normal(0.0, 0.05, size=(n_tags, n_tags)),
        emit=rng.normal(0.0, 0.05, size=(n_tags, n_obs)),
    )

    history_nll: List[float] = []

    n_train = len(train_x)
    for epoch in range(1, epochs + 1):
        grad_sum = CRFParams(
            start=np.zeros_like(params.start),
            end=np.zeros_like(params.end),
            trans=np.zeros_like(params.trans),
            emit=np.zeros_like(params.emit),
        )
        total_ll = 0.0

        for obs, gold in zip(train_x, train_y):
            grad, ll, _, _, _ = sequence_gradients(params, obs, gold)
            add_scaled_params(grad_sum, grad, 1.0)
            total_ll += ll

        # Average gradient and add L2 penalty term.
        inv_n = 1.0 / n_train
        grad_sum.start *= inv_n
        grad_sum.end *= inv_n
        grad_sum.trans *= inv_n
        grad_sum.emit *= inv_n
        l2_apply_to_gradient(grad_sum, params, l2=l2)

        # Mild learning-rate decay for smoother late-stage convergence.
        lr_t = lr / np.sqrt(1.0 + 0.03 * epoch)
        add_scaled_params(params, grad_sum, lr_t)

        avg_nll = -total_ll / n_train
        history_nll.append(float(avg_nll))

    return params, history_nll


def make_synthetic_sequence_dataset(
    n_sequences: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """Generate supervised sequence labeling data.

    Data is sampled from a small Markov process with tag-dependent observations.
    This gives labeled sequences suitable for CRF training/evaluation.
    """
    rng = np.random.default_rng(seed)

    n_tags = 3
    n_obs = 6

    start_prob = np.array([0.60, 0.30, 0.10], dtype=np.float64)
    trans_prob = np.array(
        [
            [0.85, 0.12, 0.03],
            [0.08, 0.84, 0.08],
            [0.04, 0.16, 0.80],
        ],
        dtype=np.float64,
    )
    emit_prob = np.array(
        [
            [0.56, 0.34, 0.07, 0.02, 0.01, 0.00],
            [0.03, 0.07, 0.40, 0.38, 0.10, 0.02],
            [0.00, 0.02, 0.08, 0.16, 0.38, 0.36],
        ],
        dtype=np.float64,
    )

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for _ in range(n_sequences):
        t_len = int(rng.integers(min_len, max_len + 1))
        tags = np.zeros(t_len, dtype=np.int64)
        obs = np.zeros(t_len, dtype=np.int64)

        tags[0] = rng.choice(n_tags, p=start_prob)
        obs[0] = rng.choice(n_obs, p=emit_prob[tags[0]])

        for t in range(1, t_len):
            tags[t] = rng.choice(n_tags, p=trans_prob[tags[t - 1]])
            obs[t] = rng.choice(n_obs, p=emit_prob[tags[t]])

        xs.append(obs)
        ys.append(tags)

    return xs, ys, n_tags, n_obs


def token_accuracy(params: CRFParams, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]) -> float:
    """Compute token-level accuracy using Viterbi predictions."""
    correct = 0
    total = 0
    for obs, gold in zip(xs, ys):
        pred = viterbi_decode(params, obs)
        correct += int(np.sum(pred == gold))
        total += int(gold.size)
    return float(correct / max(total, 1))


def exact_match_ratio(params: CRFParams, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]) -> float:
    """Compute exact sequence match ratio."""
    hit = 0
    for obs, gold in zip(xs, ys):
        pred = viterbi_decode(params, obs)
        hit += int(np.array_equal(pred, gold))
    return float(hit / max(len(xs), 1))


def run_consistency_checks(params: CRFParams, obs: np.ndarray) -> None:
    """Check probability identities on one sample sequence."""
    fb = forward_backward(params, obs)
    gamma, xi = marginals(params, obs, fb)

    if not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("gamma rows do not sum to 1")
    if xi.shape[0] > 0 and not np.allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-10):
        raise RuntimeError("xi slices do not sum to 1")

    if xi.shape[0] > 0:
        if not np.allclose(xi.sum(axis=2), gamma[:-1], atol=3e-9):
            raise RuntimeError("sum_j xi_t(i,j) != gamma_t(i)")
        if not np.allclose(xi.sum(axis=1), gamma[1:], atol=3e-9):
            raise RuntimeError("sum_i xi_t(i,j) != gamma_{t+1}(j)")


def main() -> None:
    print("Linear-chain CRF MVP (MATH-0324)")
    print("=" * 72)

    train_x, train_y, n_tags, n_obs = make_synthetic_sequence_dataset(
        n_sequences=260,
        min_len=8,
        max_len=13,
        seed=1324,
    )
    test_x, test_y, _, _ = make_synthetic_sequence_dataset(
        n_sequences=90,
        min_len=8,
        max_len=13,
        seed=2324,
    )

    params, nll_hist = train_crf(
        train_x,
        train_y,
        n_tags=n_tags,
        n_obs=n_obs,
        epochs=85,
        lr=0.35,
        l2=5e-4,
        seed=324,
    )

    # Verify probabilistic consistency of forward-backward marginals.
    run_consistency_checks(params, train_x[0])

    train_acc = token_accuracy(params, train_x, train_y)
    test_acc = token_accuracy(params, test_x, test_y)
    train_em = exact_match_ratio(params, train_x, train_y)
    test_em = exact_match_ratio(params, test_x, test_y)

    print(f"train sequences: {len(train_x)}, test sequences: {len(test_x)}")
    print(f"num tags: {n_tags}, num observations: {n_obs}")
    print(f"initial avg NLL: {nll_hist[0]:.6f}")
    print(f"middle  avg NLL: {nll_hist[len(nll_hist)//2]:.6f}")
    print(f"final   avg NLL: {nll_hist[-1]:.6f}")
    print(f"train token accuracy: {train_acc:.4f}")
    print(f"test  token accuracy: {test_acc:.4f}")
    print(f"train exact-match ratio: {train_em:.4f}")
    print(f"test  exact-match ratio: {test_em:.4f}")

    sample_obs = test_x[0]
    sample_gold = test_y[0]
    sample_pred = viterbi_decode(params, sample_obs)
    print("-" * 72)
    print("sample observation sequence:", sample_obs.tolist())
    print("sample gold tags        :", sample_gold.tolist())
    print("sample viterbi tags     :", sample_pred.tolist())

    if not (nll_hist[-1] < nll_hist[0]):
        raise RuntimeError("training did not lower average NLL")
    if not (test_acc > 0.78):
        raise RuntimeError("test token accuracy is unexpectedly low")

    print("=" * 72)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
