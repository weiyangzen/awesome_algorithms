"""ELECTRA minimal runnable MVP.

This demo implements a tiny Replaced Token Detection (RTD) pipeline:
1) A generator predicts masked tokens (small MLM objective).
2) A discriminator learns to detect which tokens were replaced.
3) We transfer the discriminator encoder to a sequence classification task.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    vocab_size: int = 64
    mask_token_id: int = 0
    seq_len: int = 20
    emb_dim: int = 48
    hidden_dim: int = 48
    pretrain_steps: int = 140
    batch_size: int = 64
    mask_ratio: float = 0.15
    lr_generator: float = 2e-3
    lr_discriminator: float = 2e-3
    gen_loss_weight: float = 1.0
    finetune_epochs: int = 10
    finetune_batch_size: int = 64
    finetune_lr: float = 2e-3
    train_size: int = 1400
    test_size: int = 600


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        h, _ = self.rnn(x)
        return h  # [B, L, 2H]


class TinyGenerator(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.encoder = TinyEncoder(cfg.vocab_size, cfg.emb_dim, cfg.hidden_dim)
        self.head = nn.Linear(2 * cfg.hidden_dim, cfg.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(token_ids)
        return self.head(h)  # [B, L, V]


class TinyDiscriminator(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.encoder = TinyEncoder(cfg.vocab_size, cfg.emb_dim, cfg.hidden_dim)
        self.head = nn.Linear(2 * cfg.hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(token_ids)
        return self.head(h).squeeze(-1)  # [B, L]


class SequenceClassifier(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.encoder = TinyEncoder(cfg.vocab_size, cfg.emb_dim, cfg.hidden_dim)
        self.head = nn.Linear(2 * cfg.hidden_dim, 2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(token_ids)  # [B, L, 2H]
        pooled = h.mean(dim=1)  # [B, 2H]
        return self.head(pooled)


def sample_clean_tokens(cfg: Config, batch_size: int, device: torch.device) -> torch.Tensor:
    # Reserve ID 0 for [MASK], so actual tokens are sampled from [1, vocab_size-1].
    return torch.randint(
        low=1,
        high=cfg.vocab_size,
        size=(batch_size, cfg.seq_len),
        device=device,
    )


def build_mask(cfg: Config, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.rand((batch_size, cfg.seq_len), device=device) < cfg.mask_ratio
    # Ensure every sample has at least one masked position.
    empty_rows = (~mask).all(dim=1)
    if empty_rows.any():
        row_ids = torch.where(empty_rows)[0]
        rand_pos = torch.randint(0, cfg.seq_len, size=(row_ids.numel(),), device=device)
        mask[row_ids, rand_pos] = True
    return mask


def sample_token_different_from_target(
    target: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    # Sample from [1, vocab_size-1] while excluding target token IDs.
    # target is guaranteed in [1, vocab_size-1].
    rnd = torch.randint(1, vocab_size - 1, size=target.shape, device=target.device)
    return rnd + (rnd >= target).long()


def build_corrupted_input(
    cfg: Config,
    clean: torch.Tensor,
    mask: torch.Tensor,
    gen_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = torch.multinomial(F.softmax(gen_logits[mask], dim=-1), num_samples=1).squeeze(-1)
    original_masked = clean[mask]
    same = sampled == original_masked
    if same.any():
        sampled[same] = sample_token_different_from_target(
            original_masked[same],
            vocab_size=cfg.vocab_size,
        )
    corrupted = clean.clone()
    corrupted[mask] = sampled
    # Since replacements are forced to differ from original tokens, positive labels equal mask.
    replaced_labels = mask.float()
    return corrupted, replaced_labels


def pretrain_electra(
    cfg: Config,
    generator: TinyGenerator,
    discriminator: TinyDiscriminator,
    device: torch.device,
) -> None:
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.lr_generator)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_discriminator)

    generator.train()
    discriminator.train()
    for step in range(1, cfg.pretrain_steps + 1):
        clean = sample_clean_tokens(cfg, cfg.batch_size, device)
        mask = build_mask(cfg, cfg.batch_size, device)
        masked_input = clean.clone()
        masked_input[mask] = cfg.mask_token_id

        gen_logits = generator(masked_input)
        gen_loss = F.cross_entropy(gen_logits[mask], clean[mask])

        with torch.no_grad():
            corrupted, replaced_labels = build_corrupted_input(cfg, clean, mask, gen_logits)
        disc_logits = discriminator(corrupted)
        disc_loss = F.binary_cross_entropy_with_logits(disc_logits, replaced_labels)

        total_loss = disc_loss + cfg.gen_loss_weight * gen_loss

        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)
        total_loss.backward()
        opt_g.step()
        opt_d.step()

        if step % 35 == 0 or step == 1 or step == cfg.pretrain_steps:
            with torch.no_grad():
                probs = torch.sigmoid(disc_logits)
                pred = (probs > 0.5).float()
                acc = (pred == replaced_labels).float().mean().item()
                replaced_rate = replaced_labels.mean().item()
            print(
                f"[pretrain] step={step:03d} "
                f"gen_loss={gen_loss.item():.4f} "
                f"disc_loss={disc_loss.item():.4f} "
                f"token_acc={acc:.4f} "
                f"replaced_rate={replaced_rate:.4f}"
            )


@torch.no_grad()
def evaluate_rtd(
    cfg: Config,
    generator: TinyGenerator,
    discriminator: TinyDiscriminator,
    device: torch.device,
    batches: int = 20,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    all_tp = 0.0
    all_fp = 0.0
    all_fn = 0.0
    all_correct = 0.0
    all_count = 0.0

    for _ in range(batches):
        clean = sample_clean_tokens(cfg, cfg.batch_size, device)
        mask = build_mask(cfg, cfg.batch_size, device)
        masked_input = clean.clone()
        masked_input[mask] = cfg.mask_token_id
        gen_logits = generator(masked_input)
        corrupted, labels = build_corrupted_input(cfg, clean, mask, gen_logits)

        probs = torch.sigmoid(discriminator(corrupted))
        flat_probs = probs.reshape(-1)
        flat_labels = labels.reshape(-1)
        k = int(flat_labels.sum().item())
        flat_pred = torch.zeros_like(flat_probs)
        if k > 0:
            topk_idx = torch.topk(flat_probs, k=k).indices
            flat_pred[topk_idx] = 1.0
        pred = flat_pred.reshape_as(labels)
        all_correct += (pred == labels).float().sum().item()
        all_count += float(labels.numel())
        all_tp += ((pred == 1) & (labels == 1)).float().sum().item()
        all_fp += ((pred == 1) & (labels == 0)).float().sum().item()
        all_fn += ((pred == 0) & (labels == 1)).float().sum().item()

    precision = all_tp / (all_tp + all_fp + 1e-12)
    recall = all_tp / (all_tp + all_fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    accuracy = all_correct / max(all_count, 1.0)
    return {
        "token_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_downstream_dataset(
    cfg: Config,
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = sample_clean_tokens(cfg, n_samples, device)
    left = tokens[:, : cfg.seq_len // 2].float().sum(dim=1)
    right = tokens[:, cfg.seq_len // 2 :].float().sum(dim=1)
    labels = (left > right).long()
    return tokens, labels


def train_sequence_classifier(
    cfg: Config,
    model: SequenceClassifier,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
    n = train_x.shape[0]
    steps_per_epoch = math.ceil(n / cfg.finetune_batch_size)

    for _epoch in range(cfg.finetune_epochs):
        perm = torch.randperm(n, device=train_x.device)
        x_shuf = train_x[perm]
        y_shuf = train_y[perm]

        for i in range(steps_per_epoch):
            st = i * cfg.finetune_batch_size
            ed = min((i + 1) * cfg.finetune_batch_size, n)
            xb = x_shuf[st:ed]
            yb = y_shuf[st:ed]

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate_sequence_classifier(
    model: SequenceClassifier,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> float:
    model.eval()
    logits = model(test_x)
    pred = logits.argmax(dim=1)
    return (pred == test_y).float().mean().item()


def main() -> None:
    set_seed(42)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device.type}")

    generator = TinyGenerator(cfg).to(device)
    discriminator = TinyDiscriminator(cfg).to(device)

    pretrain_electra(cfg, generator, discriminator, device)
    rtd_metrics = evaluate_rtd(cfg, generator, discriminator, device)
    print(
        "[rtd-eval] "
        + ", ".join(f"{k}={v:.4f}" for k, v in rtd_metrics.items())
    )

    train_x, train_y = build_downstream_dataset(cfg, cfg.train_size, device)
    test_x, test_y = build_downstream_dataset(cfg, cfg.test_size, device)

    pretrained_cls = SequenceClassifier(cfg).to(device)
    pretrained_cls.encoder.load_state_dict(discriminator.encoder.state_dict())
    train_sequence_classifier(cfg, pretrained_cls, train_x, train_y)
    pretrained_acc = evaluate_sequence_classifier(pretrained_cls, test_x, test_y)

    scratch_cls = SequenceClassifier(cfg).to(device)
    train_sequence_classifier(cfg, scratch_cls, train_x, train_y)
    scratch_acc = evaluate_sequence_classifier(scratch_cls, test_x, test_y)

    print(f"[downstream] accuracy_from_pretrained={pretrained_acc:.4f}")
    print(f"[downstream] accuracy_from_scratch={scratch_acc:.4f}")
    print(f"[downstream] absolute_gain={pretrained_acc - scratch_acc:+.4f}")


if __name__ == "__main__":
    main()
