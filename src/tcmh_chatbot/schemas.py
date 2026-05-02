"""Data models and schemas for the TCMH system."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single conversation turn."""
    user_id: str
    turn_id: str
    timestamp: datetime
    text: str


class ExtractionResult(BaseModel):
    """Result of NLP extraction (emotion, symptoms, triggers, mechanisms)."""
    emotion: str = "neutral"
    emotion_score: float = 0.0
    symptoms: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    crashouts: List[str] = Field(default_factory=list)
    evidence: Dict[str, List[str]] = Field(default_factory=dict)


class GraphStats(BaseModel):
    """Statistics about the temporal causal graph."""
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_degree: float = 0.0


class RiskEstimate(BaseModel):
    """Early warning risk assessment."""
    score: float = 0.0
    level: Literal["low", "medium", "high"] = "low"
    reasons: List[str] = Field(default_factory=list)


class ProcessResult(BaseModel):
    """Complete result of processing a conversation turn."""
    turn: ConversationTurn
    extraction: ExtractionResult
    risk: RiskEstimate
    graph_stats: GraphStats


class ProcessTurnRequest(BaseModel):
    """Request to process a conversation turn."""
    user_id: str
    text: str
    timestamp: Optional[datetime] = None
    turn_id: Optional[str] = None


class GraphPayload(BaseModel):
    """Serializable graph representation."""
    user_id: str
    nodes: List[Dict[str, object]] = Field(default_factory=list)
    edges: List[Dict[str, object]] = Field(default_factory=list)
