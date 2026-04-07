"""ALBERT minimal runnable MVP.

This script implements core ALBERT ideas from scratch with PyTorch:
- Factorized embedding parameterization
- Cross-layer parameter sharing
- Joint MLM + SOP pretraining objectives

The dataset is a tiny built-in corpus so the script runs offline.
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_tiny_documents() -> List[List[str]]:
    """Build a tiny but structured corpus for SOP/MLM pretraining."""
    role_topic_pairs = [
        ("engineer", "system"),
        ("doctor", "patient"),
        ("teacher", "classroom"),
        ("scientist", "experiment"),
        ("farmer", "field"),
        ("artist", "gallery"),
        ("pilot", "aircraft"),
        ("chef", "kitchen"),
        ("lawyer", "case"),
        ("analyst", "market"),
        ("nurse", "clinic"),
        ("coach", "team"),
        ("designer", "product"),
        ("writer", "article"),
        ("manager", "project"),
        ("researcher", "dataset"),
        ("planner", "city"),
        ("captain", "voyage"),
        ("chemist", "sample"),
        ("developer", "application"),
    ]

    documents: List[List[str]] = []
    for role, topic in role_topic_pairs:
        documents.append(
            [
                f"the {role} begins phase one for the {topic}",
                f"the {role} records phase two data about the {topic}",
                f"the {role} checks phase three results on the {topic}",
                f"the {role} publishes phase four report for the {topic}",
            ]
        )
    return documents


def build_vocab(documents: Sequence[Sequence[str]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for doc in documents:
        for sent in doc:
            counter.update(simple_tokenize(sent))

    vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def truncate_pair(tokens_a: List[str], tokens_b: List[str], max_len: int) -> Tuple[List[str], List[str]]:
    # Reserve 3 spots for [CLS], [SEP], [SEP].
    while len(tokens_a) + len(tokens_b) + 3 > max_len:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return tokens_a, tokens_b


def build_sop_pairs(documents: Sequence[Sequence[str]], rng: random.Random) -> List[Tuple[str, str, int]]:
    """Build positive/negative sentence pairs for SOP.

    label=0 means sentence B is the true next sentence of sentence A.
    label=1 means sentence B is replaced by a random sentence from another document.
    """
    all_sentences = [s for doc in documents for s in doc]
    pairs: List[Tuple[str, str, int]] = []

    for doc_idx, doc in enumerate(documents):
        for i in range(len(doc) - 1):
            sent_a = doc[i]
            sent_b_true = doc[i + 1]
            pairs.append((sent_a, sent_b_true, 0))

            # Negative pair: keep A and sample B from other docs.
            while True:
                sent_b_false = rng.choice(all_sentences)
                if sent_b_false not in documents[doc_idx]:
                    break
            pairs.append((sent_a, sent_b_false, 1))

    rng.shuffle(pairs)
    return pairs


def encode_pair(
    sent_a: str,
    sent_b: str,
    vocab: Dict[str, int],
    max_len: int,
) -> Tuple[List[int], List[int], List[int]]:
    pad_id = vocab["[PAD]"]
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    unk_id = vocab["[UNK]"]

    tokens_a = simple_tokenize(sent_a)
    tokens_b = simple_tokenize(sent_b)
    tokens_a, tokens_b = truncate_pair(tokens_a, tokens_b, max_len=max_len)

    token_ids = [cls_id]
    token_type_ids = [0]

    for tok in tokens_a:
        token_ids.append(vocab.get(tok, unk_id))
        token_type_ids.append(0)
    token_ids.append(sep_id)
    token_type_ids.append(0)

    for tok in tokens_b:
        token_ids.append(vocab.get(tok, unk_id))
        token_type_ids.append(1)
    token_ids.append(sep_id)
    token_type_ids.append(1)

    attention_mask = [1] * len(token_ids)

    while len(token_ids) < max_len:
        token_ids.append(pad_id)
        token_type_ids.append(0)
        attention_mask.append(0)

    return token_ids, token_type_ids, attention_mask


def apply_mlm_mask(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    mlm_probability: float,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking to a batch of token IDs."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape={tuple(input_ids.shape)}")

    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, fill_value=-100)

    special_ids = {pad_id, cls_id, sep_id}
    rows, cols = input_ids.shape

    for i in range(rows):
        for j in range(cols):
            token_id = int(input_ids[i, j].item())
            if token_id in special_ids:
                continue
            if rng.random() >= mlm_probability:
                continue

            labels[i, j] = token_id
            p = rng.random()
            if p < 0.8:
                masked_inputs[i, j] = mask_token_id
            elif p < 0.9:
                masked_inputs[i, j] = int(rng.integers(low=0, high=vocab_size))
            else:
                # Keep original token 10% of the time.
                pass

    return masked_inputs, labels


@dataclass
class AlbertBatchTensors:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    mlm_labels: torch.Tensor
    sop_labels: torch.Tensor


class AlbertPretrainDataset(Dataset[Tuple[torch.Tensor, ...]]):
    def __init__(self, batch_tensors: AlbertBatchTensors) -> None:
        self.input_ids = batch_tensors.input_ids
        self.token_type_ids = batch_tensors.token_type_ids
        self.attention_mask = batch_tensors.attention_mask
        self.mlm_labels = batch_tensors.mlm_labels
        self.sop_labels = batch_tensors.sop_labels

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.input_ids[index],
            self.token_type_ids[index],
            self.attention_mask[index],
            self.mlm_labels[index],
            self.sop_labels[index],
        )


class FactorizedEmbedding(nn.Module):
    """ALBERT embedding: token embedding in low dim then project to hidden dim."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        pad_token_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=pad_token_id,
        )
        self.embedding_projection = nn.Linear(embedding_size, hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand_as(input_ids)

        tok_embed = self.token_embeddings(input_ids)
        hidden = self.embedding_projection(tok_embed)
        hidden = hidden + self.position_embeddings(position_ids)
        hidden = hidden + self.token_type_embeddings(token_type_ids)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        return hidden


class SharedTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_ln = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_ln = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask=True means "ignore this position".
        key_padding_mask = attention_mask == 0
        attn_out, _ = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = self.attn_ln(hidden_states + self.attn_dropout(attn_out))

        ffn_out = self.ffn(hidden_states)
        hidden_states = self.ffn_ln(hidden_states + self.ffn_dropout(ffn_out))
        return hidden_states


class TinyALBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 32,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        max_position_embeddings: int = 32,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers

        self.embeddings = FactorizedEmbedding(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            pad_token_id=pad_token_id,
            dropout=dropout,
        )

        # Cross-layer sharing: reuse this single block many times.
        self.shared_layer = SharedTransformerLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        # MLM head with tied token-embedding matrix.
        self.mlm_dense = nn.Linear(hidden_size, hidden_size)
        self.mlm_act = nn.GELU()
        self.mlm_ln = nn.LayerNorm(hidden_size)
        self.mlm_to_embedding = nn.Linear(hidden_size, embedding_size)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        # SOP head from [CLS].
        self.sop_classifier = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        for _ in range(self.num_hidden_layers):
            hidden_states = self.shared_layer(hidden_states=hidden_states, attention_mask=attention_mask)

        mlm_hidden = self.mlm_ln(self.mlm_act(self.mlm_dense(hidden_states)))
        mlm_embed = self.mlm_to_embedding(mlm_hidden)
        mlm_logits = mlm_embed @ self.embeddings.token_embeddings.weight.t() + self.mlm_bias

        cls_state = hidden_states[:, 0, :]
        sop_logits = self.sop_classifier(cls_state)
        return mlm_logits, sop_logits


def build_pretrain_tensors(
    vocab: Dict[str, int],
    pairs: Sequence[Tuple[str, str, int]],
    max_len: int,
    mlm_probability: float,
    rng: np.random.Generator,
) -> AlbertBatchTensors:
    encoded_ids: List[List[int]] = []
    encoded_types: List[List[int]] = []
    encoded_masks: List[List[int]] = []
    sop_labels: List[int] = []

    for sent_a, sent_b, sop_label in pairs:
        ids, types, mask = encode_pair(sent_a, sent_b, vocab=vocab, max_len=max_len)
        encoded_ids.append(ids)
        encoded_types.append(types)
        encoded_masks.append(mask)
        sop_labels.append(sop_label)

    input_ids = torch.tensor(encoded_ids, dtype=torch.long)
    token_type_ids = torch.tensor(encoded_types, dtype=torch.long)
    attention_mask = torch.tensor(encoded_masks, dtype=torch.long)

    input_ids_masked, mlm_labels = apply_mlm_mask(
        input_ids=input_ids,
        vocab_size=len(vocab),
        mask_token_id=vocab["[MASK]"],
        pad_id=vocab["[PAD]"],
        cls_id=vocab["[CLS]"],
        sep_id=vocab["[SEP]"],
        mlm_probability=mlm_probability,
        rng=rng,
    )

    return AlbertBatchTensors(
        input_ids=input_ids_masked,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        mlm_labels=mlm_labels,
        sop_labels=torch.tensor(sop_labels, dtype=torch.long),
    )


def split_train_test(
    batch_tensors: AlbertBatchTensors,
    test_ratio: float,
    random_state: int,
) -> Tuple[AlbertBatchTensors, AlbertBatchTensors]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    indices = np.arange(batch_tensors.input_ids.shape[0])
    labels = batch_tensors.sop_labels.numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels,
    )

    def take(idx: np.ndarray) -> AlbertBatchTensors:
        idx_t = torch.tensor(idx, dtype=torch.long)
        return AlbertBatchTensors(
            input_ids=batch_tensors.input_ids[idx_t],
            token_type_ids=batch_tensors.token_type_ids[idx_t],
            attention_mask=batch_tensors.attention_mask[idx_t],
            mlm_labels=batch_tensors.mlm_labels[idx_t],
            sop_labels=batch_tensors.sop_labels[idx_t],
        )

    return take(train_idx), take(test_idx)


def build_dataloaders(
    train_tensors: AlbertBatchTensors,
    test_tensors: AlbertBatchTensors,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = AlbertPretrainDataset(train_tensors)
    test_ds = AlbertPretrainDataset(test_tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_epoch(
    model: TinyALBERT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    ce_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    ce_sop = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0
    total_mlm_correct = 0
    sop_preds_all: List[int] = []
    sop_true_all: List[int] = []

    for input_ids, token_type_ids, attention_mask, mlm_labels, sop_labels in loader:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        mlm_labels = mlm_labels.to(device)
        sop_labels = sop_labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        mlm_logits, sop_logits = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        mlm_loss = ce_mlm(mlm_logits.view(-1, model.vocab_size), mlm_labels.view(-1))
        sop_loss = ce_sop(sop_logits, sop_labels)
        loss = mlm_loss + sop_loss

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            mask_positions = mlm_labels != -100
            masked_count = int(mask_positions.sum().item())
            if masked_count > 0:
                masked_pred = mlm_logits.argmax(dim=-1)[mask_positions]
                masked_true = mlm_labels[mask_positions]
                total_mlm_correct += int((masked_pred == masked_true).sum().item())
                total_tokens += masked_count

            sop_pred = sop_logits.argmax(dim=1).cpu().numpy().tolist()
            sop_true = sop_labels.cpu().numpy().tolist()
            sop_preds_all.extend(sop_pred)
            sop_true_all.extend(sop_true)

            total_loss += float(loss.item() * input_ids.shape[0])

    avg_loss = total_loss / max(len(loader.dataset), 1)
    mlm_acc = total_mlm_correct / max(total_tokens, 1)
    sop_acc = float(accuracy_score(sop_true_all, sop_preds_all)) if sop_true_all else 0.0
    return avg_loss, mlm_acc, sop_acc


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def main() -> None:
    set_global_seed(42)
    py_rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    documents = build_tiny_documents()
    vocab = build_vocab(documents)
    pairs = build_sop_pairs(documents, rng=py_rng)

    tensors = build_pretrain_tensors(
        vocab=vocab,
        pairs=pairs,
        max_len=24,
        mlm_probability=0.15,
        rng=np_rng,
    )
    train_tensors, test_tensors = split_train_test(
        batch_tensors=tensors,
        test_ratio=0.25,
        random_state=42,
    )
    train_loader, test_loader = build_dataloaders(
        train_tensors=train_tensors,
        test_tensors=test_tensors,
        batch_size=16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyALBERT(
        vocab_size=len(vocab),
        embedding_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=24,
        type_vocab_size=2,
        pad_token_id=vocab["[PAD]"],
        dropout=0.05,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    epochs = 25
    print(f"device: {device}")
    print(f"documents={len(documents)}, sop_pairs={len(pairs)}, vocab_size={len(vocab)}")
    print(
        "ALBERT config: "
        f"embedding_size=32, hidden_size=64, shared_layers=4, heads=4, params={count_parameters(model)}"
    )

    final_test_mlm = 0.0
    final_test_sop = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_mlm_acc, train_sop_acc = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_mlm_acc, test_sop_acc = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            device=device,
        )
        final_test_mlm = test_mlm_acc
        final_test_sop = test_sop_acc

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f}, train_mlm_acc={train_mlm_acc:.4f}, train_sop_acc={train_sop_acc:.4f} | "
                f"test_loss={test_loss:.4f}, test_mlm_acc={test_mlm_acc:.4f}, test_sop_acc={test_sop_acc:.4f}"
            )

    model.eval()
    with torch.no_grad():
        sample_size = min(6, test_tensors.input_ids.shape[0])
        sample_ids = test_tensors.input_ids[:sample_size].to(device)
        sample_types = test_tensors.token_type_ids[:sample_size].to(device)
        sample_mask = test_tensors.attention_mask[:sample_size].to(device)
        sample_mlm_labels = test_tensors.mlm_labels[:sample_size]
        sample_sop_labels = test_tensors.sop_labels[:sample_size]

        sample_mlm_logits, sample_sop_logits = model(sample_ids, sample_types, sample_mask)
        sample_sop_pred = sample_sop_logits.argmax(dim=1).cpu()

    rows = []
    for i in range(sample_size):
        masked_positions = (sample_mlm_labels[i] != -100).nonzero(as_tuple=False).view(-1)
        if len(masked_positions) > 0:
            pos = int(masked_positions[0].item())
            true_id = int(sample_mlm_labels[i, pos].item())
            pred_id = int(sample_mlm_logits[i, pos].argmax().item())
        else:
            pos, true_id, pred_id = -1, -1, -1

        rows.append(
            {
                "idx": i,
                "sop_true": int(sample_sop_labels[i].item()),
                "sop_pred": int(sample_sop_pred[i].item()),
                "first_mask_pos": pos,
                "mlm_true_id": true_id,
                "mlm_pred_id": pred_id,
            }
        )

    sample_df = pd.DataFrame(rows)
    print("sample predictions:")
    print(sample_df.to_string(index=False))

    sop_baseline = 0.5
    print(f"final metrics: test_mlm_acc={final_test_mlm:.4f}, test_sop_acc={final_test_sop:.4f}")
    print(f"sop_baseline={sop_baseline:.4f}")

    if not np.isfinite(final_test_mlm) or not np.isfinite(final_test_sop):
        raise RuntimeError("evaluation metric is not finite")
    if final_test_sop < 0.70:
        raise RuntimeError(f"SOP accuracy too low: {final_test_sop:.4f} < 0.70")
    if final_test_mlm < 0.35:
        raise RuntimeError(f"MLM token accuracy too low: {final_test_mlm:.4f} < 0.35")
    if final_test_sop < sop_baseline + 0.20:
        raise RuntimeError(
            "SOP did not beat random baseline by >= 0.20: "
            f"sop={final_test_sop:.4f}, baseline={sop_baseline:.4f}"
        )

    print("All checks passed.")


if __name__ == "__main__":
    main()
