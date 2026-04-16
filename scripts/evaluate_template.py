from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tcmh_chatbot.evaluation.metrics import (
    causal_edge_precision,
    classification_metrics,
    early_warning_accuracy,
    user_understanding_rate,
)


def main() -> None:
    y_true_emotion = ["stressed", "anxious", "sad", "neutral", "sad"]
    y_pred_emotion = ["stressed", "anxious", "sad", "neutral", "anxious"]
    emotion_report = classification_metrics(y_true_emotion, y_pred_emotion)

    predicted_edges = {
        ("deadline", "overthinking", "leads_to"),
        ("overthinking", "insomnia", "worsens"),
        ("insomnia", "sad", "associated_with"),
    }
    gold_edges = {
        ("deadline", "overthinking", "leads_to"),
        ("overthinking", "insomnia", "worsens"),
    }

    causal_precision = causal_edge_precision(predicted_edges, gold_edges)
    warning_accuracy = early_warning_accuracy(
        ["low", "medium", "high", "medium"],
        ["low", "medium", "medium", "medium"],
    )
    insight_rate = user_understanding_rate([5, 4, 4, 3, 5, 4])

    print("Emotion metrics:", emotion_report)
    print("Causal edge precision:", causal_precision)
    print("Early warning accuracy:", warning_accuracy)
    print("User understanding rate:", insight_rate)


if __name__ == "__main__":
    main()
