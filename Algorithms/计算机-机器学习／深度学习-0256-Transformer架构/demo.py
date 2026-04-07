"""Minimal runnable MVP for Transformer architecture (CS-0122).

This script implements a tiny, source-transparent encoder-decoder
Transformer without using ``torch.nn.Transformer`` as a black box.

Task:
- Input sequence: random digits, e.g. [2, 7, 5, 1]
- Target sequence: reversed digits, e.g. [1, 5, 7, 2]

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PAD_ID = 0
BOS_ID = 10
EOS_ID = 11
VOCAB_SIZE = 12  # digits 1..9 + PAD/BOS/EOS


@dataclass
class ModelConfig:
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0
    max_len: int = 8


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """Manual multi-head scaled dot-product attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, L, D]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, L, H, D]
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # attn_mask is broadcastable to [B, H, Q, K]
            scores = scores.masked_fill(~attn_mask, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, v)
        merged = self._merge_heads(attended)
        return self.w_o(merged)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, memory_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class TransformerSeq2Seq(nn.Module):
    """Tiny encoder-decoder Transformer for sequence transduction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.src_embed = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.tgt_embed = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.pos_encoding = PositionalEncoding(d_model=config.d_model, max_len=config.max_len)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_proj = nn.Linear(config.d_model, VOCAB_SIZE)

    def _make_src_mask(self, src_tokens: torch.Tensor) -> torch.Tensor:
        return (src_tokens != PAD_ID).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]

    def _make_tgt_mask(self, tgt_tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = (tgt_tokens != PAD_ID).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        seq_len = tgt_tokens.shape[1]
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt_tokens.device)
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        return pad_mask & causal

    def encode(self, src_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_mask = self._make_src_mask(src_tokens)
        h = self.embed_dropout(self.pos_encoding(self.src_embed(src_tokens)))
        for layer in self.encoder_layers:
            h = layer(h, src_mask)
        return h, src_mask

    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_mask = self._make_tgt_mask(tgt_tokens)
        h = self.embed_dropout(self.pos_encoding(self.tgt_embed(tgt_tokens)))
        for layer in self.decoder_layers:
            h = layer(h, memory, tgt_mask, src_mask)
        return h

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        memory, src_mask = self.encode(src_tokens)
        decoded = self.decode(tgt_tokens, memory, src_mask)
        return self.output_proj(decoded)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_reverse_dataset(
    num_samples: int,
    min_len: int,
    max_len: int,
    rng: np.random.Generator,
) -> List[Tuple[List[int], List[int]]]:
    """Generate (source_digits, reversed_digits) pairs."""
    pairs: List[Tuple[List[int], List[int]]] = []
    for _ in range(num_samples):
        length = int(rng.integers(min_len, max_len + 1))
        seq = rng.integers(1, 10, size=length).tolist()
        pairs.append((seq, list(reversed(seq))))
    return pairs


def tensorize_pairs(
    pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert data to padded tensors for teacher forcing.

    - src: source + EOS
    - tgt_in: BOS + target
    - tgt_out: target + EOS
    """
    src_rows: List[List[int]] = []
    tgt_in_rows: List[List[int]] = []
    tgt_out_rows: List[List[int]] = []

    for src_seq, tgt_seq in pairs:
        src_tokens = list(src_seq) + [EOS_ID]
        tgt_in_tokens = [BOS_ID] + list(tgt_seq)
        tgt_out_tokens = list(tgt_seq) + [EOS_ID]

        if len(src_tokens) > max_len or len(tgt_in_tokens) > max_len or len(tgt_out_tokens) > max_len:
            raise ValueError("max_len is too small for generated samples")

        src_tokens += [PAD_ID] * (max_len - len(src_tokens))
        tgt_in_tokens += [PAD_ID] * (max_len - len(tgt_in_tokens))
        tgt_out_tokens += [PAD_ID] * (max_len - len(tgt_out_tokens))

        src_rows.append(src_tokens)
        tgt_in_rows.append(tgt_in_tokens)
        tgt_out_rows.append(tgt_out_tokens)

    return (
        torch.tensor(src_rows, dtype=torch.long),
        torch.tensor(tgt_in_rows, dtype=torch.long),
        torch.tensor(tgt_out_rows, dtype=torch.long),
    )


def train_model(
    model: TransformerSeq2Seq,
    src_train: torch.Tensor,
    tgt_in_train: torch.Tensor,
    tgt_out_train: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> List[float]:
    """Train with cross entropy and return epoch loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    rng = np.random.default_rng(seed)
    n = src_train.shape[0]
    losses: List[float] = []

    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            src_batch = src_train[idx]
            tgt_in_batch = tgt_in_train[idx]
            tgt_out_batch = tgt_out_train[idx]

            optimizer.zero_grad()
            logits = model(src_batch, tgt_in_batch)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out_batch.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
        losses.append(total_loss / max(steps, 1))
    return losses


def greedy_decode(model: TransformerSeq2Seq, src_seq: Sequence[int], max_len: int) -> List[int]:
    """Autoregressive decoding until EOS or max length."""
    model.eval()
    src_tokens = torch.tensor([list(src_seq) + [EOS_ID]], dtype=torch.long)
    generated = [BOS_ID]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tokens = torch.tensor([generated], dtype=torch.long)
            logits = model(src_tokens, tgt_tokens)
            next_token = int(torch.argmax(logits[0, -1], dim=-1).item())
            if next_token == EOS_ID:
                break
            generated.append(next_token)
            if len(generated) >= max_len:
                break
    return generated[1:]


def evaluate_exact_match(
    model: TransformerSeq2Seq,
    pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
    max_len: int,
) -> float:
    """Compute exact match ratio on held-out pairs."""
    correct = 0
    for src_seq, tgt_seq in pairs:
        pred = greedy_decode(model=model, src_seq=src_seq, max_len=max_len)
        if pred == list(tgt_seq):
            correct += 1
    return correct / len(pairs)


def main() -> None:
    print("Transformer Architecture MVP (CS-0122)")
    print("=" * 72)

    set_seed(122)
    rng = np.random.default_rng(122)

    max_content_len = 7
    max_len = max_content_len + 1  # reserve one token for BOS/EOS alignment

    train_pairs = build_reverse_dataset(num_samples=480, min_len=3, max_len=max_content_len, rng=rng)
    test_pairs = build_reverse_dataset(num_samples=120, min_len=3, max_len=max_content_len, rng=rng)
    src_train, tgt_in_train, tgt_out_train = tensorize_pairs(train_pairs, max_len=max_len)

    model = TransformerSeq2Seq(
        ModelConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=128,
            dropout=0.0,
            max_len=max_len,
        )
    )

    losses = train_model(
        model=model,
        src_train=src_train,
        tgt_in_train=tgt_in_train,
        tgt_out_train=tgt_out_train,
        epochs=70,
        batch_size=32,
        lr=2e-3,
        seed=122,
    )

    exact_match = evaluate_exact_match(model=model, pairs=test_pairs, max_len=max_len)

    print(f"train_samples: {len(train_pairs)}")
    print(f"test_samples: {len(test_pairs)}")
    print(f"max_len: {max_len}")
    print("-" * 72)
    print(f"initial_loss: {losses[0]:.4f}")
    print(f"final_loss:   {losses[-1]:.4f}")
    print(f"exact_match:  {exact_match:.3f}")
    print("-" * 72)
    print("sample predictions:")
    for i, (src_seq, tgt_seq) in enumerate(test_pairs[:5], start=1):
        pred_seq = greedy_decode(model=model, src_seq=src_seq, max_len=max_len)
        print(f"  case {i}: src={src_seq} pred={pred_seq} tgt={list(tgt_seq)}")

    assert losses[0] > losses[-1], "loss did not decrease"
    assert exact_match >= 0.75, "exact match is unexpectedly low"
    print("All checks passed.")


if __name__ == "__main__":
    main()
