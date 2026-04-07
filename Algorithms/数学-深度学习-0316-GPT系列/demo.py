"""Minimal runnable MVP for GPT series (MATH-0316).

This script implements a tiny character-level decoder-only Transformer (GPT-style)
from source-level components:
- token + position embeddings
- masked multi-head self-attention
- residual blocks with MLP
- next-token autoregressive training

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Configuration for a tiny GPT model."""

    vocab_size: int
    block_size: int = 48
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention with explicit causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(emb_dim, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        att_scores = att_scores.masked_fill(~mask, float("-inf"))

        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        out = att_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN GPT block: LN -> Attention -> residual; LN -> MLP -> residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Tiny decoder-only Transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"seq_len={seq_len} exceeds block_size={self.config.block_size}."
            )

        pos = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, self.config.vocab_size),
                targets.view(batch_size * seq_len),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive text generation."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducible MVP runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_toy_corpus() -> str:
    """Build a tiny corpus describing GPT concepts for char-level modeling."""
    lines = [
        "gpt is a decoder only transformer model.",
        "it predicts the next token from previous context.",
        "causal self attention blocks future tokens with a mask.",
        "stacked blocks mix attention and feed forward layers.",
        "language modeling minimizes next token cross entropy.",
        "after training, autoregressive decoding generates text.",
        "scaling data and parameters improves capability.",
    ]
    # Repeat to make a stable tiny dataset without external files.
    return "\n".join(lines * 90)


def build_vocab(text: str) -> Tuple[Dict[str, int], List[str]]:
    """Character-level vocabulary."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = list(chars)
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    """Encode text to integer token ids."""
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(token_ids: List[int], itos: List[str]) -> str:
    """Decode integer token ids to text."""
    return "".join(itos[i] for i in token_ids)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous training chunks."""
    if len(data) <= block_size:
        raise ValueError("Dataset too short for configured block_size")

    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int = 20,
) -> Dict[str, float]:
    """Estimate average train/val losses."""
    model.eval()
    out: Dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses: List[float] = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("Loss should not be None during evaluation")
            losses.append(float(loss.item()))
        out[split] = float(np.mean(losses))
    model.train()
    return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    set_seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = build_toy_corpus()
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    config = GPTConfig(vocab_size=len(itos), block_size=48, n_embd=96, n_head=4, n_layer=2, dropout=0.0)
    model = TinyGPT(config).to(device)

    batch_size = 32
    max_steps = 260
    eval_interval = 65

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    initial = estimate_loss(model, train_data, val_data, batch_size, config.block_size, device)
    print(f"device={device.type} vocab_size={len(itos)}")
    print(f"trainable_params={count_parameters(model)}")
    print("initial_loss", {k: round(v, 4) for k, v in initial.items()})

    for step in range(1, max_steps + 1):
        xb, yb = get_batch(train_data, batch_size, config.block_size, device)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Loss should not be None during training")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == 1:
            print(f"step={step:03d} train_loss={loss.item():.4f}")

    final = estimate_loss(model, train_data, val_data, batch_size, config.block_size, device)
    print("final_loss", {k: round(v, 4) for k, v in final.items()})

    if not final["val"] < initial["val"]:
        raise AssertionError(
            f"Validation loss did not improve: initial={initial['val']:.4f}, final={final['val']:.4f}"
        )

    prompt = "gpt "
    prompt_ids = [stoi[ch] for ch in prompt]
    context = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated_ids = model.generate(context, max_new_tokens=180, temperature=0.9)[0].tolist()
    generated_text = decode(generated_ids, itos)

    print("sample_generation_start")
    print(generated_text)
    print("sample_generation_end")


if __name__ == "__main__":
    main()
