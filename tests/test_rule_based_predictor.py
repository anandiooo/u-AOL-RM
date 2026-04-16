from __future__ import annotations

from tcmh_chatbot.core.schemas import ExtractionResult, GraphStats
from tcmh_chatbot.prediction.rule_based_predictor import RuleBasedRiskPredictor


def test_rule_based_predictor_flags_medium_or_high_risk() -> None:
    predictor = RuleBasedRiskPredictor()
    extraction = ExtractionResult(
        emotion="sad",
        emotion_score=0.9,
        symptoms=["insomnia", "low mood"],
        triggers=["deadline"],
        crashouts=["overthinking"],
    )
    graph_stats = GraphStats(node_count=10, edge_count=15, density=0.28, avg_degree=3.0)

    risk = predictor.predict(extraction, graph_stats)

    assert risk.level in {"medium", "high"}
    assert risk.score >= 0.35
