from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    user_id: str
    turn_id: str
    timestamp: datetime
    text: str


class ExtractionResult(BaseModel):
    emotion: str = "neutral"
    emotion_score: float = 0.0
    symptoms: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    crashouts: List[str] = Field(default_factory=list)
    evidence: Dict[str, List[str]] = Field(default_factory=dict)


class GraphStats(BaseModel):
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_degree: float = 0.0


class RiskEstimate(BaseModel):
    score: float = 0.0
    level: Literal["low", "medium", "high"] = "low"
    reasons: List[str] = Field(default_factory=list)


class ProcessResult(BaseModel):
    turn: ConversationTurn
    extraction: ExtractionResult
    risk: RiskEstimate
    graph_stats: GraphStats


class ProcessTurnRequest(BaseModel):
    user_id: str
    text: str
    timestamp: Optional[datetime] = None
    turn_id: Optional[str] = None


class GraphPayload(BaseModel):
    user_id: str
    nodes: List[Dict[str, object]] = Field(default_factory=list)
    edges: List[Dict[str, object]] = Field(default_factory=list)
