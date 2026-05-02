"""Temporal Personal Causal Graph building and visualization."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

try:
    from pyvis.network import Network
except Exception:  # pragma: no cover - optional dependency fallback
    Network = None

from tcmh_chatbot.schemas import ConversationTurn, ExtractionResult, GraphStats


# Node colors for visualization
NODE_COLORS = {
    "trigger": {"background": "#fee2e2", "border": "#dc2626"},
    "mechanism": {"background": "#fef3c7", "border": "#d97706"},
    "symptom": {"background": "#dbeafe", "border": "#2563eb"},
    "emotion": {"background": "#ede9fe", "border": "#7c3aed"},
    "unknown": {"background": "#f1f5f9", "border": "#64748b"},
}


def _slug(value: str) -> str:
    """Convert a string to a URL-safe slug."""
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

    def _find_existing_node(self, graph: nx.DiGraph, node_type: str, label: str) -> str | None:
        """Return the ID of an existing node with the same type and label, if any."""
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("node_type") == node_type and attrs.get("label") == label:
                return node_id
        return None

    def _add_nodes(
        self,
        graph: nx.DiGraph,
        turn: ConversationTurn,
        labels: List[str],
        node_type: str,
    ) -> List[str]:
        node_ids: List[str] = []
        for index, label in enumerate(labels):
            # Reuse an existing node with the same type and label
            existing = self._find_existing_node(graph, node_type, label)
            if existing:
                # Update timestamp to the latest turn
                graph.nodes[existing]["timestamp"] = turn.timestamp.isoformat()
                graph.nodes[existing]["turn_id"] = turn.turn_id
                node_ids.append(existing)
            else:
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


class XAIVisualizer:
    """Creates a visual explanation artifact from a user causal graph."""

    def render_html(self, graph: nx.DiGraph, output_path: Path, title: str = "Temporal Personal Causal Graph") -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if Network is None:
            self._render_fallback_html(graph, output_path, title)
            return output_path

        network = Network(height="720px", width="100%", directed=True, bgcolor="#f8fafc", font_color="#0f172a")
        network.barnes_hut(gravity=-25000, central_gravity=0.15, spring_length=250)

        for node_id, attrs in graph.nodes(data=True):
            node_type = str(attrs.get("node_type", "unknown"))
            label = str(attrs.get("label", node_id))
            timestamp = str(attrs.get("timestamp", ""))
            tooltip = f"type: {node_type}<br>time: {timestamp}"
            network.add_node(
                node_id,
                label=label,
                title=tooltip,
                color=NODE_COLORS.get(node_type, NODE_COLORS["unknown"]),
            )

        for source, target, attrs in graph.edges(data=True):
            relation = str(attrs.get("relation", "related"))
            weight = float(attrs.get("weight", 1.0))
            lag_hours = float(attrs.get("lag_hours", 0.0))
            tooltip = f"relation: {relation}<br>weight: {weight}<br>lag_hours: {lag_hours}"
            network.add_edge(
                source,
                target,
                title=tooltip,
                label=relation,
                value=max(weight * 3.0, 1.0),
                arrows="to",
            )

        network.set_options(
            """
            {
              "edges": {
                "smooth": {
                  "enabled": true,
                  "type": "dynamic",
                  "roundness": 0.5
                },
                "color": {
                  "color": "#94a3b8",
                  "highlight": "#475569",
                  "hover": "#64748b"
                },
                "font": {
                  "size": 15,
                  "face": "Tahoma",
                  "align": "horizontal",
                  "strokeWidth": 3,
                  "strokeColor": "#f8fafc",
                  "color": "#334155"
                },
                "width": 2
              },
              "nodes": {
                "font": {
                  "size": 16,
                  "face": "Tahoma",
                  "color": "#0f172a"
                },
                "shape": "box",
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "shadow": {
                  "enabled": true,
                  "color": "rgba(0,0,0,0.15)",
                  "size": 10,
                  "x": 3,
                  "y": 3
                },
                "margin": 10
              },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -30000,
                  "centralGravity": 0.1,
                  "springLength": 250,
                  "springConstant": 0.05
                },
                "stabilization": {
                  "iterations": 200
                }
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 200
              }
            }
            """
        )

        network.write_html(str(output_path), open_browser=False, notebook=False)
        return output_path

    @staticmethod
    def write_json(graph_payload: Dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    @staticmethod
    def _render_fallback_html(graph: nx.DiGraph, output_path: Path, title: str) -> None:
        payload = {
            "nodes": [
                {
                    "id": node_id,
                    "label": attrs.get("label"),
                    "node_type": attrs.get("node_type"),
                    "timestamp": attrs.get("timestamp"),
                }
                for node_id, attrs in graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "relation": attrs.get("relation"),
                    "weight": attrs.get("weight"),
                    "lag_hours": attrs.get("lag_hours"),
                }
                for source, target, attrs in graph.edges(data=True)
            ],
        }

        html = (
            "<!doctype html>\n"
            "<html><head><meta charset='utf-8'><title>"
            + title
            + "</title></head><body>"
            + f"<h2>{title}</h2><pre>{json.dumps(payload, ensure_ascii=False, indent=2)}</pre>"
            + "</body></html>"
        )
        output_path.write_text(html, encoding="utf-8")
