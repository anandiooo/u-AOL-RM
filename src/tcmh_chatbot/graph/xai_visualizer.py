from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import networkx as nx

try:
    from pyvis.network import Network
except Exception:  # pragma: no cover - optional dependency fallback
    Network = None


NODE_COLORS = {
    "trigger": {"background": "#fee2e2", "border": "#dc2626"},
    "crashout": {"background": "#fef3c7", "border": "#d97706"},
    "symptom": {"background": "#dbeafe", "border": "#2563eb"},
    "emotion": {"background": "#ede9fe", "border": "#7c3aed"},
    "unknown": {"background": "#f1f5f9", "border": "#64748b"},
}


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
