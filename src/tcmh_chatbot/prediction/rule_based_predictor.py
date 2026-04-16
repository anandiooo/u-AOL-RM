from __future__ import annotations

from typing import Dict, List, Tuple

from tcmh_chatbot.core.schemas import ExtractionResult, GraphStats, RiskEstimate


DEFAULT_SYMPTOM_WEIGHTS = {
    "insomnia": 0.26,
    "lelah": 0.16,
    "hilang fokus": 0.18,
    "low mood": 0.22,
}

DEFAULT_TRIGGER_WEIGHTS = {
    "deadline": 0.18,
    "ujian": 0.16,
    "tekanan akademik": 0.20,
    "konflik keluarga": 0.20,
    "masalah finansial": 0.18,
    "kesepian": 0.16,
}

DEFAULT_EMOTION_WEIGHTS = {
    "neutral": 0.05,
    "positive": 0.0,
    "stressed": 0.15,
    "anxious": 0.19,
    "sad": 0.21,
    "hopeless": 0.30,
}


class RuleBasedRiskPredictor:
    """Baseline early warning scoring model based on weighted evidence."""

    def __init__(self, rules: Dict[str, Dict[str, float]] | None = None) -> None:
        rules = rules or {}

        self.symptom_weights = {**DEFAULT_SYMPTOM_WEIGHTS, **rules.get("symptom_weights", {})}
        self.trigger_weights = {**DEFAULT_TRIGGER_WEIGHTS, **rules.get("trigger_weights", {})}
        self.emotion_weights = {**DEFAULT_EMOTION_WEIGHTS, **rules.get("emotion_weights", {})}

        thresholds = rules.get("thresholds", {})
        self.medium_threshold = float(thresholds.get("medium", 0.35))
        self.high_threshold = float(thresholds.get("high", 0.65))

    def predict(self, extraction: ExtractionResult, graph_stats: GraphStats) -> RiskEstimate:
        score = 0.0
        reasons: List[Tuple[float, str]] = []

        symptom_total_weight = 0.0
        for symptom in extraction.symptoms:
            weight = float(self.symptom_weights.get(symptom, 0.08))
            score += weight
            symptom_total_weight += weight
        if extraction.symptoms:
            reasons.append((symptom_total_weight, f"symptom: {', '.join(extraction.symptoms)}"))

        trigger_total_weight = 0.0
        for trigger in extraction.triggers:
            weight = float(self.trigger_weights.get(trigger, 0.07))
            score += weight
            trigger_total_weight += weight
        if extraction.triggers:
            reasons.append((trigger_total_weight, f"trigger: {', '.join(extraction.triggers)}"))

        emotion_weight = float(self.emotion_weights.get(extraction.emotion, 0.05))
        emotion_contribution = emotion_weight * max(0.5, extraction.emotion_score)
        score += emotion_contribution
        reasons.append((emotion_contribution, f"emotion:{extraction.emotion}"))

        graph_contribution = min(0.2, graph_stats.density * 0.5 + graph_stats.edge_count * 0.005)
        score += graph_contribution
        reasons.append((graph_contribution, "graph:temporal_density"))

        score = min(score, 1.0)

        if score >= self.high_threshold:
            level = "high"
        elif score >= self.medium_threshold:
            level = "medium"
        else:
            level = "low"

        ranked_reasons = [reason for _, reason in sorted(reasons, key=lambda item: item[0], reverse=True) if _ > 0.0]
        return RiskEstimate(score=round(score, 4), level=level, reasons=ranked_reasons[:4])
