"""Self-attention mechanism MVP (single-head, PyTorch, no training).

Run:
    uv run python demo.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionConfig:
    seq_len: int = 6
    d_model: int = 8
    d_k: int = 8
    d_v: int = 8
    use_causal_mask: bool = True
    seed: int = 20260407


@dataclass
class AttentionResult:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    scores: torch.Tensor
    probs: torch.Tensor
    output: torch.Tensor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def build_input_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
    """Construct deterministic token embeddings with slight positional bias."""
    x = torch.randn(seq_len, d_model, dtype=torch.float32)
    pos = torch.linspace(0.0, 1.0, steps=seq_len, dtype=torch.float32).unsqueeze(1)
    return x + 0.2 * pos


def build_projection_matrices(d_model: int, d_k: int, d_v: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Small scale keeps logits numerically stable and easy to read.
    scale = 0.5
    w_q = scale * torch.randn(d_model, d_k, dtype=torch.float32)
    w_k = scale * torch.randn(d_model, d_k, dtype=torch.float32)
    w_v = scale * torch.randn(d_model, d_v, dtype=torch.float32)
    return w_q, w_k, w_v


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """True means masked (cannot attend)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]
    scores = (q @ k.transpose(0, 1)) / math.sqrt(float(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    output = probs @ v
    return output, scores, probs


def self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    mask: torch.Tensor | None,
) -> AttentionResult:
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    output, scores, probs = scaled_dot_product_attention(q, k, v, mask)
    return AttentionResult(q=q, k=k, v=v, scores=scores, probs=probs, output=output)


def self_attention_reference_loops(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Loop-based reference implementation for correctness cross-check."""
    seq_len = q.shape[0]
    d_k = q.shape[-1]
    output = torch.zeros(seq_len, v.shape[-1], dtype=q.dtype)

    for i in range(seq_len):
        logits = torch.empty(seq_len, dtype=q.dtype)
        for j in range(seq_len):
            if mask is not None and bool(mask[i, j]):
                logits[j] = torch.finfo(q.dtype).min
            else:
                logits[j] = torch.dot(q[i], k[j]) / math.sqrt(float(d_k))
        weights = torch.softmax(logits, dim=0)

        context_i = torch.zeros(v.shape[-1], dtype=q.dtype)
        for j in range(seq_len):
            context_i += weights[j] * v[j]
        output[i] = context_i

    return output


def format_tensor(t: torch.Tensor, decimals: int = 4) -> str:
    rounded = torch.round(t * (10**decimals)) / (10**decimals)
    return str(rounded)


def run_checks(result: AttentionResult, mask: torch.Tensor | None) -> None:
    seq_len = result.probs.shape[0]

    # 1) Attention rows should sum to 1.
    row_sums = result.probs.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones(seq_len), atol=1e-6):
        raise AssertionError("Attention probabilities do not sum to 1 row-wise.")

    # 2) Causal mask should force upper-triangular probabilities to ~0.
    if mask is not None:
        upper = torch.triu(result.probs, diagonal=1)
        if float(upper.abs().max()) > 1e-7:
            raise AssertionError("Causal mask leakage detected in upper-triangular region.")

    # 3) Compare vectorized result with loop reference.
    ref_output = self_attention_reference_loops(result.q, result.k, result.v, mask)
    max_err = float((result.output - ref_output).abs().max())
    if max_err > 1e-5:
        raise AssertionError(f"Vectorized attention mismatch with loop reference: {max_err:.6e}")


def main() -> None:
    cfg = AttentionConfig()
    set_seed(cfg.seed)

    x = build_input_embeddings(cfg.seq_len, cfg.d_model)
    w_q, w_k, w_v = build_projection_matrices(cfg.d_model, cfg.d_k, cfg.d_v)
    mask = make_causal_mask(cfg.seq_len) if cfg.use_causal_mask else None

    result = self_attention(x, w_q, w_k, w_v, mask)
    run_checks(result, mask)

    print("Self-Attention MVP (single-head)")
    print(f"seq_len={cfg.seq_len}, d_model={cfg.d_model}, d_k={cfg.d_k}, d_v={cfg.d_v}")
    print(f"causal_mask={cfg.use_causal_mask}, seed={cfg.seed}")
    print()
    print("Input embedding shape:", tuple(x.shape))
    print("Q/K/V shape:", tuple(result.q.shape), tuple(result.k.shape), tuple(result.v.shape))
    print("Attention score shape:", tuple(result.scores.shape))
    print("Attention prob shape:", tuple(result.probs.shape))
    print("Output shape:", tuple(result.output.shape))
    print()
    print("Attention probabilities (rounded):")
    print(format_tensor(result.probs))
    print()
    print("Output vectors (rounded):")
    print(format_tensor(result.output))
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
