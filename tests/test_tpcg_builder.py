from __future__ import annotations

from datetime import datetime

from tcmh_chatbot.core.schemas import ConversationTurn, ExtractionResult
from tcmh_chatbot.graph.tpcg_builder import TPCGBuilder


def test_tpcg_builder_adds_nodes_and_edges() -> None:
    builder = TPCGBuilder(max_link_hours=72)

    turn = ConversationTurn(
        user_id="student_01",
        turn_id="t1",
        timestamp=datetime(2026, 4, 1, 20, 0, 0),
        text="Aku stres karena deadline dan jadi susah tidur.",
    )
    extraction = ExtractionResult(
        emotion="stressed",
        emotion_score=0.9,
        triggers=["deadline"],
        crashouts=["overthinking"],
        symptoms=["insomnia"],
    )

    builder.add_turn(turn, extraction)
    stats = builder.graph_stats("student_01")

    assert stats.node_count >= 4
    assert stats.edge_count >= 3
