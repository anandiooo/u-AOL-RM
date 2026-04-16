from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from tcmh_chatbot.core.config import load_model_config, load_risk_rules, project_root
from tcmh_chatbot.core.schemas import (
    ConversationTurn,
    ExtractionResult,
    ProcessResult,
    ProcessTurnRequest,
)
from tcmh_chatbot.graph.tpcg_builder import TPCGBuilder
from tcmh_chatbot.graph.xai_visualizer import XAIVisualizer
from tcmh_chatbot.nlp.emotion_detector import EmotionDetector
from tcmh_chatbot.nlp.symptom_trigger_extractor import SymptomTriggerExtractor
from tcmh_chatbot.prediction.rule_based_predictor import RuleBasedRiskPredictor


class TemporalCausalChatbot:
    """Coordinates NLP extraction, temporal causal graph updates, and risk scoring."""

    def __init__(self, model_config: Dict[str, Any] | None = None, risk_rules: Dict[str, Any] | None = None) -> None:
        self.model_config = model_config or load_model_config()
        self.risk_rules = risk_rules or load_risk_rules()

        emotion_keywords = self.model_config.get("emotion_keywords", {})
        lexicons = self.model_config.get("lexicons", {})
        max_link_hours = float(self.model_config.get("time_window_hours", 168))

        self.emotion_detector = EmotionDetector(emotion_keywords=emotion_keywords)
        self.entity_extractor = SymptomTriggerExtractor(lexicons=lexicons)
        self.graph_builder = TPCGBuilder(max_link_hours=max_link_hours)
        self.risk_predictor = RuleBasedRiskPredictor(rules=self.risk_rules)
        self.visualizer = XAIVisualizer()

    def process_turn(self, request: ProcessTurnRequest) -> ProcessResult:
        timestamp = request.timestamp or datetime.utcnow()
        turn_id = request.turn_id or f"turn_{uuid.uuid4().hex[:10]}"

        turn = ConversationTurn(
            user_id=request.user_id,
            turn_id=turn_id,
            timestamp=timestamp,
            text=request.text,
        )

        emotion, emotion_score, emotion_evidence = self.emotion_detector.detect(turn.text)
        entity_payload = self.entity_extractor.extract(turn.text)

        extraction = ExtractionResult(
            emotion=emotion,
            emotion_score=emotion_score,
            symptoms=entity_payload["symptoms"],
            triggers=entity_payload["triggers"],
            crashouts=entity_payload["crashouts"],
            evidence={
                "emotion": emotion_evidence.get(emotion, []),
                "symptoms": entity_payload["evidence"]["symptoms"],
                "triggers": entity_payload["evidence"]["triggers"],
                "crashouts": entity_payload["evidence"]["crashouts"],
            },
        )

        self.graph_builder.add_turn(turn, extraction)
        graph_stats = self.graph_builder.graph_stats(turn.user_id)
        risk = self.risk_predictor.predict(extraction, graph_stats)

        return ProcessResult(turn=turn, extraction=extraction, risk=risk, graph_stats=graph_stats)

    def get_user_graph(self, user_id: str) -> Dict[str, Any]:
        return self.graph_builder.to_dict(user_id)

    def export_user_graph_json(self, user_id: str, output_path: Path | None = None) -> Path:
        graph_payload = self.graph_builder.to_dict(user_id)
        destination = output_path or project_root() / "outputs" / f"tpcg_{user_id}.json"
        return self.visualizer.write_json(graph_payload, destination)

    def export_user_xai_html(self, user_id: str, output_path: Path | None = None) -> Path:
        graph = self.graph_builder.get_graph(user_id)
        destination = output_path or project_root() / "outputs" / f"tpcg_{user_id}.html"
        return self.visualizer.render_html(graph, destination)
