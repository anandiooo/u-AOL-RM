from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    average: str = "macro",
) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "accuracy": float(round(accuracy, 4)),
    }


def causal_edge_precision(
    predicted_edges: Iterable[Tuple[str, str, str]],
    gold_edges: Iterable[Tuple[str, str, str]],
) -> float:
    predicted_set = set(predicted_edges)
    gold_set = set(gold_edges)

    if not predicted_set:
        return 0.0

    true_positive = len(predicted_set.intersection(gold_set))
    return round(true_positive / len(predicted_set), 4)


def early_warning_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return round(float(accuracy_score(y_true, y_pred)), 4)


def user_understanding_rate(scores: Sequence[float], threshold: float = 4.0) -> float:
    if not scores:
        return 0.0
    passed = sum(score >= threshold for score in scores)
    return round(passed / len(scores), 4)
