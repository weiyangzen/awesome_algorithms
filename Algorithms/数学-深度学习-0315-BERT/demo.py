"""Minimal runnable MVP for BERT (MATH-0315).

This script implements a tiny, source-transparent BERT pretraining demo:
- input representation: token + position + segment embeddings
- encoder: bidirectional TransformerEncoder
- objectives: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)

No interactive input is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]


@dataclass
class Vocab:
    """Simple vocabulary container."""

    stoi: Dict[str, int]
    itos: List[str]


@dataclass
class PretrainBatch:
    """All tensors needed by BERT pretraining."""

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    mlm_labels: torch.Tensor
    nsp_labels: torch.Tensor


class TinyBERT(nn.Module):
    """A small BERT-style model for educational MVP runs."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.seg_embed = nn.Embedding(2, hidden_size)

        self.embed_norm = nn.LayerNorm(hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.nsp_head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.max_len}")

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_ids = pos_ids.expand(batch_size, seq_len)

        h = self.token_embed(input_ids) + self.pos_embed(pos_ids) + self.seg_embed(token_type_ids)
        h = self.embed_norm(h)
        h = self.embed_dropout(h)

        # True means "ignore" for Transformer key padding mask.
        key_padding_mask = attention_mask == 0
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        mlm_logits = self.mlm_head(h)
        cls_h = h[:, 0, :]
        nsp_logits = self.nsp_head(cls_h)
        return mlm_logits, nsp_logits


def simple_tokenize(text: str) -> List[str]:
    """Lowercase + regex tokenization for a tiny MVP."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def build_toy_documents() -> List[List[str]]:
    """Construct a tiny corpus split by documents and sentences."""
    return [
        [
            "BERT uses bidirectional attention for language understanding",
            "The model is pretrained with masked language modeling",
            "Next sentence prediction is another pretraining objective",
            "Fine tuning adapts the encoder to downstream tasks",
        ],
        [
            "Transformers replace recurrence with self attention",
            "Token embeddings are summed with positional embeddings",
            "Multi head attention captures relations between words",
            "Feed forward layers refine contextual representations",
        ],
        [
            "Machine learning systems need robust evaluation",
            "Validation datasets estimate generalization quality",
            "Optimization with AdamW often speeds up training",
            "Gradient clipping can stabilize deep neural networks",
        ],
        [
            "Natural language processing covers text classification",
            "Question answering needs context aware reasoning",
            "Sentence pairs can be used for entailment tasks",
            "Pretraining enables transfer with limited labels",
        ],
    ]


def build_vocab(documents: Sequence[Sequence[str]]) -> Vocab:
    """Build vocabulary from corpus tokens with fixed special tokens."""
    words: List[str] = []
    for doc in documents:
        for sent in doc:
            words.extend(simple_tokenize(sent))

    uniq_words = sorted(set(words))
    itos = SPECIAL_TOKENS + uniq_words
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def encode_documents(
    documents: Sequence[Sequence[str]],
    stoi: Dict[str, int],
    unk_token: str = "[UNK]",
) -> List[List[List[int]]]:
    """Convert documents to token-id documents."""
    unk_id = stoi[unk_token]
    encoded: List[List[List[int]]] = []
    for doc in documents:
        encoded_doc: List[List[int]] = []
        for sent in doc:
            token_ids = [stoi.get(tok, unk_id) for tok in simple_tokenize(sent)]
            encoded_doc.append(token_ids)
        encoded.append(encoded_doc)
    return encoded


def make_nsp_pairs(
    encoded_docs: Sequence[Sequence[Sequence[int]]],
    rng: np.random.Generator,
) -> List[Tuple[List[int], List[int], int]]:
    """Create positive/negative NSP pairs.

    Returns list of (sentence_a_ids, sentence_b_ids, nsp_label), where
    nsp_label=1 means B is the true next sentence, 0 means random sentence.
    """
    pairs: List[Tuple[List[int], List[int], int]] = []

    for doc_idx, doc in enumerate(encoded_docs):
        if len(doc) < 2:
            continue

        for i in range(len(doc) - 1):
            sent_a = list(doc[i])
            sent_b_true = list(doc[i + 1])
            pairs.append((sent_a, sent_b_true, 1))

            # Build one negative sample for each positive sample.
            other_doc_indices = [j for j in range(len(encoded_docs)) if j != doc_idx and len(encoded_docs[j]) > 0]
            neg_doc_idx = int(rng.choice(other_doc_indices))
            neg_doc = encoded_docs[neg_doc_idx]
            neg_sent_idx = int(rng.integers(0, len(neg_doc)))
            sent_b_neg = list(neg_doc[neg_sent_idx])
            pairs.append((sent_a, sent_b_neg, 0))

    rng.shuffle(pairs)
    return pairs


def truncate_sentence_pair(a_ids: List[int], b_ids: List[int], max_tokens_without_special: int) -> None:
    """In-place truncation to keep total length under budget."""
    while len(a_ids) + len(b_ids) > max_tokens_without_special:
        if len(a_ids) >= len(b_ids):
            a_ids.pop()
        else:
            b_ids.pop()


def apply_mlm_mask(
    input_ids: List[int],
    special_ids: Sequence[int],
    mask_id: int,
    vocab_size: int,
    rng: np.random.Generator,
    mask_ratio: float = 0.15,
) -> Tuple[List[int], List[int]]:
    """Apply BERT-style MLM masking.

    - 15% of non-special tokens are selected.
    - Selected tokens: 80% -> [MASK], 10% -> random token, 10% -> unchanged.
    - Unselected positions have label -100 and are ignored by CE loss.
    """
    masked = list(input_ids)
    labels = [-100] * len(input_ids)
    special_id_set = set(special_ids)

    candidate_positions = [
        idx for idx, token_id in enumerate(input_ids) if token_id not in special_id_set
    ]
    if not candidate_positions:
        return masked, labels

    n_mask = max(1, int(round(len(candidate_positions) * mask_ratio)))
    n_mask = min(n_mask, len(candidate_positions))
    picked = rng.choice(candidate_positions, size=n_mask, replace=False)

    non_special_low = len(SPECIAL_TOKENS)
    for pos in picked.tolist():
        labels[pos] = input_ids[pos]
        draw = float(rng.random())
        if draw < 0.8:
            masked[pos] = mask_id
        elif draw < 0.9:
            random_token = int(rng.integers(non_special_low, vocab_size))
            masked[pos] = random_token
        else:
            # Keep unchanged token.
            pass

    return masked, labels


def build_pretrain_batch(
    vocab: Vocab,
    encoded_docs: Sequence[Sequence[Sequence[int]]],
    max_len: int,
    seed: int,
) -> PretrainBatch:
    """Create tensorized MLM+NSP pretraining batch."""
    rng = np.random.default_rng(seed)
    pairs = make_nsp_pairs(encoded_docs=encoded_docs, rng=rng)

    cls_id = vocab.stoi["[CLS]"]
    sep_id = vocab.stoi["[SEP]"]
    pad_id = vocab.stoi["[PAD]"]
    mask_id = vocab.stoi["[MASK]"]

    all_input_ids: List[List[int]] = []
    all_token_type_ids: List[List[int]] = []
    all_attention_mask: List[List[int]] = []
    all_mlm_labels: List[List[int]] = []
    all_nsp_labels: List[int] = []

    for sent_a, sent_b, nsp_label in pairs:
        a_ids = list(sent_a)
        b_ids = list(sent_b)

        # Reserve 3 slots for [CLS], [SEP], [SEP].
        truncate_sentence_pair(a_ids, b_ids, max_tokens_without_special=max_len - 3)

        input_ids = [cls_id] + a_ids + [sep_id] + b_ids + [sep_id]
        token_type_ids = [0] * (len(a_ids) + 2) + [1] * (len(b_ids) + 1)
        attention_mask = [1] * len(input_ids)

        masked_ids, mlm_labels = apply_mlm_mask(
            input_ids=input_ids,
            special_ids=[cls_id, sep_id, pad_id],
            mask_id=mask_id,
            vocab_size=len(vocab.itos),
            rng=rng,
            mask_ratio=0.15,
        )

        pad_count = max_len - len(masked_ids)
        if pad_count < 0:
            raise RuntimeError("Negative pad_count encountered. Check truncation logic.")

        masked_ids.extend([pad_id] * pad_count)
        token_type_ids.extend([0] * pad_count)
        attention_mask.extend([0] * pad_count)
        mlm_labels.extend([-100] * pad_count)

        all_input_ids.append(masked_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        all_mlm_labels.append(mlm_labels)
        all_nsp_labels.append(nsp_label)

    return PretrainBatch(
        input_ids=torch.tensor(all_input_ids, dtype=torch.long),
        token_type_ids=torch.tensor(all_token_type_ids, dtype=torch.long),
        attention_mask=torch.tensor(all_attention_mask, dtype=torch.long),
        mlm_labels=torch.tensor(all_mlm_labels, dtype=torch.long),
        nsp_labels=torch.tensor(all_nsp_labels, dtype=torch.long),
    )


def train_tiny_bert(
    model: TinyBERT,
    batch: PretrainBatch,
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 3e-3,
) -> Dict[str, List[float]]:
    """Train with AdamW and return epoch loss traces."""
    dataset = TensorDataset(
        batch.input_ids,
        batch.token_type_ids,
        batch.attention_mask,
        batch.mlm_labels,
        batch.nsp_labels,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_losses: List[float] = []
    mlm_losses: List[float] = []
    nsp_losses: List[float] = []

    model.train()
    for _ in range(epochs):
        sum_total = 0.0
        sum_mlm = 0.0
        sum_nsp = 0.0
        n_batches = 0

        for input_ids, token_type_ids, attention_mask, mlm_labels, nsp_labels in loader:
            optimizer.zero_grad()

            mlm_logits, nsp_logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

            mlm_loss = F.cross_entropy(
                mlm_logits.reshape(-1, mlm_logits.shape[-1]),
                mlm_labels.reshape(-1),
                ignore_index=-100,
            )
            nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
            total_loss = mlm_loss + nsp_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sum_total += float(total_loss.item())
            sum_mlm += float(mlm_loss.item())
            sum_nsp += float(nsp_loss.item())
            n_batches += 1

        total_losses.append(sum_total / n_batches)
        mlm_losses.append(sum_mlm / n_batches)
        nsp_losses.append(sum_nsp / n_batches)

    return {
        "total": total_losses,
        "mlm": mlm_losses,
        "nsp": nsp_losses,
    }


def evaluate_nsp_accuracy(model: TinyBERT, batch: PretrainBatch) -> float:
    """Evaluate NSP accuracy on the toy pretraining pairs."""
    model.eval()
    with torch.no_grad():
        _, nsp_logits = model(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            attention_mask=batch.attention_mask,
        )
        pred = torch.argmax(nsp_logits, dim=1)
        acc = (pred == batch.nsp_labels).float().mean().item()
    return float(acc)


def inspect_mlm_predictions(
    model: TinyBERT,
    batch: PretrainBatch,
    vocab: Vocab,
    max_items: int = 5,
) -> List[Tuple[int, str, str]]:
    """Return (position, true_token, predicted_token) on masked positions."""
    model.eval()
    with torch.no_grad():
        mlm_logits, _ = model(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            attention_mask=batch.attention_mask,
        )

    sample_idx = 0
    labels = batch.mlm_labels[sample_idx]
    preds = torch.argmax(mlm_logits[sample_idx], dim=-1)

    outputs: List[Tuple[int, str, str]] = []
    for pos in range(labels.shape[0]):
        true_id = int(labels[pos].item())
        if true_id == -100:
            continue
        pred_id = int(preds[pos].item())
        outputs.append((pos, vocab.itos[true_id], vocab.itos[pred_id]))
        if len(outputs) >= max_items:
            break
    return outputs


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible MVP behavior."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    print("BERT tiny pretraining MVP (MATH-0315)")
    print("=" * 72)

    set_seed(315)

    documents = build_toy_documents()
    vocab = build_vocab(documents)
    encoded_docs = encode_documents(documents=documents, stoi=vocab.stoi)

    max_len = 20
    batch = build_pretrain_batch(vocab=vocab, encoded_docs=encoded_docs, max_len=max_len, seed=315)

    model = TinyBERT(
        vocab_size=len(vocab.itos),
        max_len=max_len,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        ffn_mult=4,
        dropout=0.0,
    )

    history = train_tiny_bert(
        model=model,
        batch=batch,
        epochs=80,
        batch_size=8,
        lr=3e-3,
    )

    nsp_acc = evaluate_nsp_accuracy(model=model, batch=batch)
    mlm_examples = inspect_mlm_predictions(model=model, batch=batch, vocab=vocab, max_items=5)

    print(f"num_pretrain_pairs: {batch.input_ids.shape[0]}")
    print(f"vocab_size: {len(vocab.itos)}")
    print(f"sequence_length: {batch.input_ids.shape[1]}")
    print("-" * 72)
    print(f"initial_total_loss: {history['total'][0]:.4f}")
    print(f"final_total_loss:   {history['total'][-1]:.4f}")
    print(f"final_mlm_loss:     {history['mlm'][-1]:.4f}")
    print(f"final_nsp_loss:     {history['nsp'][-1]:.4f}")
    print(f"nsp_accuracy:       {nsp_acc:.3f}")
    print("-" * 72)
    print("MLM sample predictions (position, true -> pred):")
    for pos, true_tok, pred_tok in mlm_examples:
        print(f"  pos {pos:2d}: {true_tok:>15s} -> {pred_tok:<15s}")

    # Basic quality gates for a deterministic toy run.
    assert history["total"][0] > history["total"][-1], "training loss did not decrease"
    assert nsp_acc >= 0.70, "NSP accuracy is unexpectedly low"


if __name__ == "__main__":
    main()
