from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import networkx as nx

from tcmh_chatbot.core.schemas import ConversationTurn, ExtractionResult, GraphStats


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "item"


class TPCGBuilder:
    """Builds a per-user temporal causal graph from extracted entities."""

    def __init__(self, max_link_hours: float = 168.0) -> None:
        self.max_link_hours = max_link_hours
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _get_graph(self, user_id: str) -> nx.DiGraph:
        if user_id not in self._graphs:
            self._graphs[user_id] = nx.DiGraph()
        return self._graphs[user_id]

    @staticmethod
    def _add_edge(
        graph: nx.DiGraph,
        source: str,
        target: str,
        relation: str,
        weight: float,
        lag_hours: float,
    ) -> None:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] = round(float(graph[source][target].get("weight", 0.0)) + weight, 3)
            return

        graph.add_edge(
            source,
            target,
            relation=relation,
            weight=round(weight, 3),
            lag_hours=round(lag_hours, 2),
        )

    @staticmethod
    def _make_node_id(turn_id: str, node_type: str, label: str, index: int) -> str:
        return f"{turn_id}:{node_type}:{_slug(label)}:{index}"

    def _add_nodes(
        self,
        graph: nx.DiGraph,
        turn: ConversationTurn,
        labels: List[str],
        node_type: str,
    ) -> List[str]:
        node_ids: List[str] = []
        for index, label in enumerate(labels):
            node_id = self._make_node_id(turn.turn_id, node_type, label, index)
            graph.add_node(
                node_id,
                node_type=node_type,
                label=label,
                timestamp=turn.timestamp.isoformat(),
                turn_id=turn.turn_id,
            )
            node_ids.append(node_id)
        return node_ids

    def add_turn(self, turn: ConversationTurn, extraction: ExtractionResult) -> None:
        graph = self._get_graph(turn.user_id)

        trigger_nodes = self._add_nodes(graph, turn, extraction.triggers, "trigger")
        crashout_nodes = self._add_nodes(graph, turn, extraction.crashouts, "crashout")
        symptom_nodes = self._add_nodes(graph, turn, extraction.symptoms, "symptom")
        emotion_node = self._add_nodes(graph, turn, [extraction.emotion], "emotion")[0]

        for trigger_node in trigger_nodes:
            for crashout_node in crashout_nodes:
                self._add_edge(graph, trigger_node, crashout_node, "leads_to", 1.0, 0.0)

        for crashout_node in crashout_nodes:
            for symptom_node in symptom_nodes:
                self._add_edge(graph, crashout_node, symptom_node, "worsens", 1.0, 0.0)

        if trigger_nodes and symptom_nodes and not crashout_nodes:
            for trigger_node in trigger_nodes:
                for symptom_node in symptom_nodes:
                    self._add_edge(graph, trigger_node, symptom_node, "direct_effect", 0.9, 0.0)

        for symptom_node in symptom_nodes:
            self._add_edge(graph, symptom_node, emotion_node, "associated_with", 1.0, 0.0)

        if not symptom_nodes and (trigger_nodes or crashout_nodes):
            for source_node in trigger_nodes + crashout_nodes:
                self._add_edge(graph, source_node, emotion_node, "associated_with", 0.6, 0.0)

        self._add_temporal_edges(turn.user_id, graph, symptom_nodes + [emotion_node])
        self._update_history(turn.user_id, graph, trigger_nodes + crashout_nodes + symptom_nodes + [emotion_node])

    def _add_temporal_edges(self, user_id: str, graph: nx.DiGraph, target_nodes: List[str]) -> None:
        for target_node in target_nodes:
            target_meta = graph.nodes[target_node]
            target_time = datetime.fromisoformat(str(target_meta["timestamp"]))
            target_label = str(target_meta.get("label", ""))

            for record in self._history[user_id]:
                lag_hours = (target_time - record["timestamp"]).total_seconds() / 3600.0
                if lag_hours <= 0 or lag_hours > self.max_link_hours:
                    continue

                if record["node_id"] == target_node:
                    continue

                # Keep temporal edges sparse: connect if same label or from trigger signals.
                if record["node_type"] != "trigger" and record["label"] != target_label:
                    continue

                decay = max(0.15, 1.0 - (lag_hours / self.max_link_hours))
                self._add_edge(
                    graph,
                    record["node_id"],
                    target_node,
                    "temporal_influence",
                    0.4 * decay,
                    lag_hours,
                )

    def _update_history(self, user_id: str, graph: nx.DiGraph, node_ids: List[str]) -> None:
        for node_id in node_ids:
            metadata = graph.nodes[node_id]
            self._history[user_id].append(
                {
                    "node_id": node_id,
                    "node_type": metadata.get("node_type"),
                    "label": metadata.get("label"),
                    "timestamp": datetime.fromisoformat(str(metadata["timestamp"])),
                }
            )

    def has_graph(self, user_id: str) -> bool:
        return user_id in self._graphs and self._graphs[user_id].number_of_nodes() > 0

    def get_graph(self, user_id: str) -> nx.DiGraph:
        return self._get_graph(user_id)

    def graph_stats(self, user_id: str) -> GraphStats:
        graph = self._get_graph(user_id)
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        density = float(nx.density(graph)) if node_count > 1 else 0.0
        avg_degree = (
            sum(dict(graph.degree()).values()) / node_count
            if node_count > 0
            else 0.0
        )

        return GraphStats(
            node_count=node_count,
            edge_count=edge_count,
            density=round(density, 4),
            avg_degree=round(avg_degree, 4),
        )

    def to_dict(self, user_id: str) -> Dict[str, Any]:
        graph = self._get_graph(user_id)

        nodes = [
            {
                "id": node_id,
                "label": str(attrs.get("label", "")),
                "node_type": str(attrs.get("node_type", "unknown")),
                "timestamp": str(attrs.get("timestamp", "")),
                "turn_id": str(attrs.get("turn_id", "")),
            }
            for node_id, attrs in graph.nodes(data=True)
        ]

        edges = [
            {
                "source": source,
                "target": target,
                "relation": str(attrs.get("relation", "related")),
                "weight": float(attrs.get("weight", 1.0)),
                "lag_hours": float(attrs.get("lag_hours", 0.0)),
            }
            for source, target, attrs in graph.edges(data=True)
        ]

        return {"user_id": user_id, "nodes": nodes, "edges": edges}
