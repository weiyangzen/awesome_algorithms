"""Minimal runnable MVP for Multinomial Naive Bayes text classification.

This script provides:
1) A transparent source-level implementation of Multinomial Naive Bayes.
2) A comparison against sklearn.naive_bayes.MultinomialNB.
3) Non-interactive, reproducible output with assertions as quality gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


@dataclass
class ManualMultinomialNB:
    """Container for manually trained Multinomial Naive Bayes parameters."""

    classes_: np.ndarray
    class_log_prior_: np.ndarray
    feature_log_prob_: np.ndarray
    alpha: float


def build_synthetic_corpus(seed: int = 224) -> tuple[List[str], np.ndarray, Dict[int, str]]:
    """Build a reproducible synthetic corpus with three topics.

    Each document is a whitespace-tokenized sentence so CountVectorizer can be
    used directly without extra tokenizers.
    """
    rng = np.random.default_rng(seed)

    label_to_name = {0: "sports", 1: "finance", 2: "tech"}
    topic_vocab = {
        0: ["team", "match", "goal", "coach", "league", "season", "stadium", "attack", "defense"],
        1: ["market", "stock", "risk", "return", "fund", "bank", "trade", "policy", "capital"],
        2: ["model", "data", "python", "network", "cloud", "chip", "vector", "training", "inference"],
    }
    common_vocab = ["today", "update", "analysis", "report", "trend", "focus"]

    texts: List[str] = []
    labels: List[int] = []

    docs_per_class = 70
    for cls in sorted(label_to_name):
        other_classes = [c for c in topic_vocab if c != cls]
        for _ in range(docs_per_class):
            core_tokens = rng.choice(topic_vocab[cls], size=9, replace=True).tolist()
            common_tokens = rng.choice(common_vocab, size=3, replace=True).tolist()

            noise_cls = int(rng.choice(other_classes))
            noise_tokens = rng.choice(topic_vocab[noise_cls], size=2, replace=True).tolist()

            doc_tokens = core_tokens + common_tokens + noise_tokens
            rng.shuffle(doc_tokens)
            texts.append(" ".join(doc_tokens))
            labels.append(cls)

    return texts, np.asarray(labels, dtype=int), label_to_name


def fit_manual_multinomial_nb(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> ManualMultinomialNB:
    """Fit Multinomial Naive Bayes from count matrix and labels.

    Formula:
    - P(c) = N_c / N
    - P(w_j | c) = (N_{c,j} + alpha) / (sum_j N_{c,j} + alpha * V)
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have the same number of rows")
    if np.min(x_train.data, initial=0.0) < 0.0:
        raise ValueError("MultinomialNB requires non-negative feature counts")

    classes = np.unique(y_train)
    n_classes = classes.shape[0]
    n_features = x_train.shape[1]

    class_count = np.zeros(n_classes, dtype=np.float64)
    feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

    for idx, cls in enumerate(classes):
        class_mask = y_train == cls
        class_count[idx] = float(np.sum(class_mask))
        class_feature_sum = x_train[class_mask].sum(axis=0)
        feature_count[idx] = np.asarray(class_feature_sum).ravel()

    class_log_prior = np.log(class_count / float(np.sum(class_count)))

    smoothed = feature_count + alpha
    smoothed_norm = smoothed.sum(axis=1, keepdims=True)
    feature_log_prob = np.log(smoothed) - np.log(smoothed_norm)

    return ManualMultinomialNB(
        classes_=classes,
        class_log_prior_=class_log_prior,
        feature_log_prob_=feature_log_prob,
        alpha=alpha,
    )


def manual_joint_log_likelihood(model: ManualMultinomialNB, x: sparse.csr_matrix) -> np.ndarray:
    """Compute log P(c) + sum_j x_j log P(w_j|c) for each sample and class."""
    return x @ model.feature_log_prob_.T + model.class_log_prior_


def manual_predict(model: ManualMultinomialNB, x: sparse.csr_matrix) -> np.ndarray:
    """Predict classes by argmax of joint log-likelihood."""
    jll = manual_joint_log_likelihood(model, x)
    class_indices = np.argmax(jll, axis=1)
    return model.classes_[class_indices]


def top_tokens_per_class(
    feature_log_prob: np.ndarray,
    feature_names: Sequence[str],
    label_to_name: Dict[int, str],
    top_k: int = 8,
) -> pd.DataFrame:
    """Return top-k tokens with highest P(token | class)."""
    rows = []
    for cls in sorted(label_to_name):
        order = np.argsort(feature_log_prob[cls])[::-1][:top_k]
        tokens = [feature_names[i] for i in order]
        probs = np.exp(feature_log_prob[cls, order])
        rows.append(
            {
                "class": label_to_name[cls],
                "top_tokens": ", ".join(tokens),
                "token_probs": ", ".join(f"{p:.3f}" for p in probs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    texts, labels, label_to_name = build_synthetic_corpus(seed=224)

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=224,
        stratify=labels,
    )

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
    x_train = vectorizer.fit_transform(x_train_text).tocsr()
    x_test = vectorizer.transform(x_test_text).tocsr()

    alpha = 1.0
    manual_model = fit_manual_multinomial_nb(x_train=x_train, y_train=y_train, alpha=alpha)
    manual_pred = manual_predict(manual_model, x_test)

    sklearn_model = MultinomialNB(alpha=alpha)
    sklearn_model.fit(x_train, y_train)
    sklearn_pred = sklearn_model.predict(x_test)

    manual_acc = float(accuracy_score(y_test, manual_pred))
    sklearn_acc = float(accuracy_score(y_test, sklearn_pred))
    manual_f1 = float(f1_score(y_test, manual_pred, average="macro"))
    sklearn_f1 = float(f1_score(y_test, sklearn_pred, average="macro"))
    agreement = float(np.mean(manual_pred == sklearn_pred))

    metrics_table = pd.DataFrame(
        {
            "model": ["manual_multinomial_nb", "sklearn_multinomial_nb"],
            "accuracy": [manual_acc, sklearn_acc],
            "f1_macro": [manual_f1, sklearn_f1],
        }
    )

    conf = confusion_matrix(y_test, manual_pred, labels=[0, 1, 2])
    conf_df = pd.DataFrame(
        conf,
        index=[f"true_{label_to_name[i]}" for i in [0, 1, 2]],
        columns=[f"pred_{label_to_name[i]}" for i in [0, 1, 2]],
    )

    top_token_df = top_tokens_per_class(
        feature_log_prob=manual_model.feature_log_prob_,
        feature_names=vectorizer.get_feature_names_out(),
        label_to_name=label_to_name,
        top_k=8,
    )

    assert x_train.shape[0] > 0 and x_test.shape[0] > 0, "empty split"
    assert x_train.shape[1] >= 20, "vocabulary unexpectedly small"
    assert np.min(x_train.data, initial=0.0) >= 0.0, "negative counts detected"
    assert agreement >= 0.95, "manual and sklearn predictions diverge too much"
    assert manual_acc >= 0.85, "manual model accuracy below expected baseline"

    print("Multinomial Naive Bayes MVP")
    print(
        f"train_samples={x_train.shape[0]}, test_samples={x_test.shape[0]}, "
        f"vocab_size={x_train.shape[1]}, alpha={alpha}"
    )
    print(f"manual_vs_sklearn_agreement={agreement:.4f}")
    print()

    print("metric_summary=")
    print(metrics_table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
    print()

    print("manual_confusion_matrix=")
    print(conf_df.to_string())
    print()

    print("manual_top_tokens=")
    print(top_token_df.to_string(index=False))
    print()

    report = classification_report(
        y_test,
        manual_pred,
        target_names=[label_to_name[i] for i in [0, 1, 2]],
        digits=4,
        zero_division=0,
    )
    print("manual_classification_report=")
    print(report)


if __name__ == "__main__":
    main()
