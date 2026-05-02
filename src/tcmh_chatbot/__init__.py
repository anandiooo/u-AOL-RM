"""Temporal Causal Mental Health Chatbot - A proof-of-concept system for explainable AI-driven mental health support."""

from tcmh_chatbot.engine import TemporalCausalChatbot
from tcmh_chatbot.schemas import (
    ConversationTurn,
    ExtractionResult,
    GraphPayload,
    GraphStats,
    ProcessResult,
    ProcessTurnRequest,
    RiskEstimate,
)

__version__ = "0.1.0"

__all__ = [
    "TemporalCausalChatbot",
    "ConversationTurn",
    "ExtractionResult",
    "GraphPayload",
    "GraphStats",
    "ProcessResult",
    "ProcessTurnRequest",
    "RiskEstimate",
]
