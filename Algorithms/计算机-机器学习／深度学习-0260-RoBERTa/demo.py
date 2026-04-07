"""RoBERTa minimal runnable MVP.

This script demonstrates the core RoBERTa ideas with a tiny offline setup:
1) Transformer encoder pretraining with MLM only (no NSP).
2) Dynamic masking applied on-the-fly for every batch.
3) A lightweight downstream transfer check with a linear probe.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
TOPIC_ROLE_PAIRS: List[Tuple[str, str]] = [
    ("market", "analyst"),
    ("health", "doctor"),
    ("sports", "coach"),
    ("technology", "engineer"),
    ("education", "teacher"),
    ("travel", "pilot"),
]


@dataclass
class Config:
    max_len: int = 28
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_size: int = 128
    dropout: float = 0.1
    mlm_probability: float = 0.15
    batch_size: int = 24
    lr: float = 2e-3
    pretrain_epochs: int = 18
    probe_batch_size: int = 64


class TextTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.attention_mask[index]


class TinyRoBERTaForMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        ff_size: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.mlm_dense = nn.Linear(hidden_size, hidden_size)
        self.mlm_norm = nn.LayerNorm(hidden_size)
        self.mlm_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        # Weight tying: decoder shares weights with token embeddings.
        self.mlm_decoder.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        key_padding_mask = attention_mask == 0
        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mlm_hidden = self.mlm_dense(hidden)
        mlm_hidden = F.gelu(mlm_hidden)
        mlm_hidden = self.mlm_norm(mlm_hidden)
        logits = self.mlm_decoder(mlm_hidden) + self.mlm_bias
        return logits, hidden


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_pretrain_dataframe() -> pd.DataFrame:
    docs: List[dict[str, str]] = []

    for topic, role in TOPIC_ROLE_PAIRS:
        for item_id in range(1, 8):
            s1 = f"the {role} reviews {topic} plan step one for case {item_id}"
            s2 = f"the {role} updates {topic} report step two with evidence {item_id}"
            s3 = f"the {role} validates {topic} result step three before release {item_id}"
            s4 = f"the {role} archives {topic} notes after step three for case {item_id}"
            docs.append({"topic": topic, "text": f"{s1} {s2} {s3} {s4}"})

    return pd.DataFrame(docs)


def build_vocab(texts: Sequence[str], min_freq: int = 1) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for token, count in counter.most_common():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], List[int]]:
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    pad_id = vocab["[PAD]"]
    unk_id = vocab["[UNK]"]

    tokens = simple_tokenize(text)
    tokens = tokens[: max_len - 2]

    token_ids = [cls_id] + [vocab.get(tok, unk_id) for tok in tokens] + [sep_id]
    attention_mask = [1] * len(token_ids)

    while len(token_ids) < max_len:
        token_ids.append(pad_id)
        attention_mask.append(0)

    return token_ids, attention_mask


def encode_many(texts: Sequence[str], vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ids_list: List[List[int]] = []
    mask_list: List[List[int]] = []
    for text in texts:
        ids, mask = encode_text(text, vocab=vocab, max_len=max_len)
        ids_list.append(ids)
        mask_list.append(mask)

    input_ids = torch.tensor(ids_list, dtype=torch.long)
    attention_mask = torch.tensor(mask_list, dtype=torch.long)
    return input_ids, attention_mask


def apply_dynamic_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_token_id: int,
    special_token_ids: set[int],
    mlm_probability: float,
    valid_replace_ids: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, fill_value=-100)

    rows, cols = input_ids.shape
    for i in range(rows):
        valid_positions = [
            j
            for j in range(cols)
            if int(attention_mask[i, j].item()) == 1 and int(input_ids[i, j].item()) not in special_token_ids
        ]
        if not valid_positions:
            continue

        selected = rng.random(len(valid_positions)) < mlm_probability
        if not selected.any():
            selected[rng.integers(low=0, high=len(valid_positions))] = True

        for pos, flag in zip(valid_positions, selected):
            if not flag:
                continue
            original_id = int(input_ids[i, pos].item())
            labels[i, pos] = original_id

            coin = rng.random()
            if coin < 0.8:
                masked_inputs[i, pos] = mask_token_id
            elif coin < 0.9:
                replacement = int(rng.choice(valid_replace_ids))
                masked_inputs[i, pos] = replacement
            else:
                # Keep the original token 10% of the time.
                pass

    return masked_inputs, labels


def run_mlm_epoch(
    model: TinyRoBERTaForMLM,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    config: Config,
    vocab_size: int,
    mask_token_id: int,
    special_token_ids: set[int],
    valid_replace_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    total_batches = 0
    masked_correct = 0
    masked_total = 0
    entropy_values: List[float] = []

    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        masked_input_ids, mlm_labels = apply_dynamic_masking(
            input_ids,
            attention_mask,
            mask_token_id=mask_token_id,
            special_token_ids=special_token_ids,
            mlm_probability=config.mlm_probability,
            valid_replace_ids=valid_replace_ids,
            rng=rng,
        )
        masked_input_ids = masked_input_ids.to(device)
        mlm_labels = mlm_labels.to(device)

        logits, _ = model(masked_input_ids, attention_mask)
        loss = criterion(logits.reshape(-1, vocab_size), mlm_labels.reshape(-1))

        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        with torch.no_grad():
            active = mlm_labels != -100
            active_count = int(active.sum().item())
            if active_count > 0:
                preds = logits.argmax(dim=-1)
                masked_correct += int((preds[active] == mlm_labels[active]).sum().item())
                masked_total += active_count
                if not train_mode:
                    probs = torch.softmax(logits[active], dim=-1)
                    sampled_probs = probs[: min(128, probs.shape[0])].detach().cpu().numpy()
                    entropy_values.extend(entropy(sampled_probs, axis=1).tolist())

    avg_loss = total_loss / max(total_batches, 1)
    masked_acc = masked_correct / max(masked_total, 1)
    perplexity = float(np.exp(min(avg_loss, 20.0)))
    avg_entropy = float(np.mean(entropy_values)) if entropy_values else float("nan")

    return {
        "loss": avg_loss,
        "masked_acc": masked_acc,
        "perplexity": perplexity,
        "masked_entropy": avg_entropy,
    }


def build_role_topic_probe_dataframe(n_samples: int, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    topics = [topic for topic, _ in TOPIC_ROLE_PAIRS]
    role_by_topic = {topic: role for topic, role in TOPIC_ROLE_PAIRS}
    roles = [role for _, role in TOPIC_ROLE_PAIRS]
    rows: List[dict[str, int | str]] = []

    for _ in range(n_samples):
        topic = rng.choice(topics)
        is_matched = int(rng.random() < 0.5)
        if is_matched:
            role = role_by_topic[topic]
        else:
            wrong_roles = [role for role in roles if role != role_by_topic[topic]]
            role = rng.choice(wrong_roles)

        text = (
            f"the {role} reviews {topic} plan step one before update "
            f"the {role} updates {topic} report step two with notes "
            f"the {role} validates {topic} result step three after review"
        )
        rows.append({"text": text, "label": is_matched})

    return pd.DataFrame(rows)


@torch.no_grad()
def extract_cls_embeddings(
    model: TinyRoBERTaForMLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    dataset = TextTensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_vecs: List[np.ndarray] = []

    for ids_batch, mask_batch in loader:
        ids_batch = ids_batch.to(device)
        mask_batch = mask_batch.to(device)
        _, hidden = model(ids_batch, mask_batch)
        cls_vec = hidden[:, 0, :].detach().cpu().numpy()
        all_vecs.append(cls_vec)

    return np.concatenate(all_vecs, axis=0)


def run_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    clf = LogisticRegression(max_iter=400, random_state=42)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return {
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
    }


def main() -> None:
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    pretrain_df = build_pretrain_dataframe()
    vocab = build_vocab(pretrain_df["text"].tolist())
    vocab_size = len(vocab)

    input_ids, attention_mask = encode_many(pretrain_df["text"].tolist(), vocab=vocab, max_len=cfg.max_len)
    train_idx, val_idx = train_test_split(
        np.arange(len(pretrain_df)),
        test_size=0.25,
        random_state=42,
        stratify=pretrain_df["topic"].values,
    )

    train_dataset = TextTensorDataset(input_ids[train_idx], attention_mask[train_idx])
    val_dataset = TextTensorDataset(input_ids[val_idx], attention_mask[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = TinyRoBERTaForMLM(
        vocab_size=vocab_size,
        max_len=cfg.max_len,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_size=cfg.ff_size,
        dropout=cfg.dropout,
        pad_id=vocab["[PAD]"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    special_token_ids = {vocab[tok] for tok in SPECIAL_TOKENS}
    valid_replace_ids = np.array([idx for idx in range(vocab_size) if idx not in special_token_ids], dtype=np.int64)
    train_rng = np.random.default_rng(2026)
    eval_rng = np.random.default_rng(2027)

    print(f"device={device.type}")
    print(
        f"pretrain_samples={len(pretrain_df)} train={len(train_idx)} val={len(val_idx)} "
        f"vocab_size={vocab_size}"
    )
    print(
        "config="
        f"hidden={cfg.hidden_size}, layers={cfg.num_layers}, heads={cfg.num_heads}, "
        f"max_len={cfg.max_len}, mlm_prob={cfg.mlm_probability}, epochs={cfg.pretrain_epochs}"
    )

    final_val_metrics: dict[str, float] = {}
    for epoch in range(1, cfg.pretrain_epochs + 1):
        train_metrics = run_mlm_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            config=cfg,
            vocab_size=vocab_size,
            mask_token_id=vocab["[MASK]"],
            special_token_ids=special_token_ids,
            valid_replace_ids=valid_replace_ids,
            rng=train_rng,
        )
        val_metrics = run_mlm_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            config=cfg,
            vocab_size=vocab_size,
            mask_token_id=vocab["[MASK]"],
            special_token_ids=special_token_ids,
            valid_replace_ids=valid_replace_ids,
            rng=eval_rng,
        )
        final_val_metrics = val_metrics
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['masked_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['masked_acc']:.4f} "
            f"val_ppl={val_metrics['perplexity']:.3f} val_entropy={val_metrics['masked_entropy']:.3f}"
        )

    probe_df = build_role_topic_probe_dataframe(n_samples=420, seed=17)
    probe_input_ids, probe_attention_mask = encode_many(probe_df["text"].tolist(), vocab=vocab, max_len=cfg.max_len)
    labels = probe_df["label"].to_numpy(dtype=np.int64)

    p_train_idx, p_test_idx = train_test_split(
        np.arange(len(probe_df)),
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    train_ids = probe_input_ids[p_train_idx]
    train_mask = probe_attention_mask[p_train_idx]
    test_ids = probe_input_ids[p_test_idx]
    test_mask = probe_attention_mask[p_test_idx]
    y_train = labels[p_train_idx]
    y_test = labels[p_test_idx]

    pretrained_train_x = extract_cls_embeddings(model, train_ids, train_mask, cfg.probe_batch_size, device)
    pretrained_test_x = extract_cls_embeddings(model, test_ids, test_mask, cfg.probe_batch_size, device)

    random_model = TinyRoBERTaForMLM(
        vocab_size=vocab_size,
        max_len=cfg.max_len,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_size=cfg.ff_size,
        dropout=cfg.dropout,
        pad_id=vocab["[PAD]"],
    ).to(device)
    random_train_x = extract_cls_embeddings(random_model, train_ids, train_mask, cfg.probe_batch_size, device)
    random_test_x = extract_cls_embeddings(random_model, test_ids, test_mask, cfg.probe_batch_size, device)

    pretrained_metrics = run_linear_probe(pretrained_train_x, y_train, pretrained_test_x, y_test)
    random_metrics = run_linear_probe(random_train_x, y_train, random_test_x, y_test)

    report_df = pd.DataFrame(
        [
            {
                "encoder": "pretrained_roberta",
                "accuracy": pretrained_metrics["accuracy"],
                "macro_f1": pretrained_metrics["macro_f1"],
            },
            {
                "encoder": "random_init",
                "accuracy": random_metrics["accuracy"],
                "macro_f1": random_metrics["macro_f1"],
            },
        ]
    )

    gain = pretrained_metrics["accuracy"] - random_metrics["accuracy"]
    print("\n[linear-probe] role-topic compatibility classification")
    print(report_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"accuracy_gain_pretrained_minus_random={gain:.4f}")

    if final_val_metrics["masked_acc"] <= 0.12:
        raise RuntimeError("MLM masked accuracy is too low; pretraining likely failed.")
    if not np.isfinite(final_val_metrics["masked_entropy"]):
        raise RuntimeError("Validation entropy is not finite.")

    print("All checks passed.")


if __name__ == "__main__":
    main()
