from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tcmh_chatbot.chatbot.engine import TemporalCausalChatbot
from tcmh_chatbot.core.schemas import ProcessTurnRequest


def _init_state() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = TemporalCausalChatbot()
    if "turn_history" not in st.session_state:
        st.session_state.turn_history = []
    if "active_user_id" not in st.session_state:
        st.session_state.active_user_id = "student_01"


def _process_turn(user_id: str, text: str, timestamp: str | None = None, turn_id: str | None = None) -> Dict[str, Any]:
    request = ProcessTurnRequest(
        user_id=user_id,
        text=text,
        timestamp=timestamp,
        turn_id=turn_id,
    )
    result = st.session_state.engine.process_turn(request)
    payload = result.model_dump(mode="json")
    st.session_state.turn_history.append(payload)
    return payload


def _load_sample_conversations(user_id: str) -> int:
    sample_path = PROJECT_ROOT / "data" / "sample" / "sample_conversations.jsonl"
    if not sample_path.exists():
        return 0

    counter = 0
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _process_turn(
                user_id=user_id,
                text=row["text"],
                timestamp=row.get("timestamp"),
                turn_id=row.get("turn_id"),
            )
            counter += 1
    return counter


def _latest_for_user(user_id: str) -> Dict[str, Any] | None:
    for item in reversed(st.session_state.turn_history):
        if item["turn"]["user_id"] == user_id:
            return item
    return None


def _history_for_user(user_id: str) -> List[Dict[str, Any]]:
    return [item for item in st.session_state.turn_history if item["turn"]["user_id"] == user_id]


def _risk_color(level: str) -> str:
    if level == "high":
        return "#dc2626"
    if level == "medium":
        return "#f59e0b"
    return "#16a34a"


def _render_dashboard() -> None:
    st.set_page_config(page_title="Temporal Causal Mental Health Chatbot", layout="wide")
    st.title("Temporal Causal Mental Health Chatbot")
    st.caption("A proactive, causal-aware mental health support system with Explainable AI.")

    with st.sidebar:
        st.header("Session Settings")
        user_id = st.text_input("User ID", value=st.session_state.active_user_id)
        st.session_state.active_user_id = user_id.strip() or "student_01"

        if st.button("Load Sample Conversations", use_container_width=True):
            loaded = _load_sample_conversations(st.session_state.active_user_id)
            if loaded:
                st.success(f"Loaded {loaded} sample turns")
            else:
                st.warning("Sample conversation file not found")

        if st.button("Reset Session", use_container_width=True):
            st.session_state.engine = TemporalCausalChatbot()
            st.session_state.turn_history = []
            st.rerun()

    # Structure into columns for a dashboard feel
    main_col, side_col = st.columns([2, 1])

    with main_col:
        st.subheader("Conversation")
        history = _history_for_user(st.session_state.active_user_id)

        # Display chat history
        for item in history:
            with st.chat_message("user"):
                st.write(item["turn"]["text"])
            with st.chat_message("assistant"):
                emo = item["extraction"]["emotion"]
                risk_lvl = item["risk"]["level"]
                st.markdown(f"*Detected emotion:* **{emo.upper()}** | *Risk:* **{risk_lvl.upper()}**")

        # Chat input
        if prompt := st.chat_input("Tell me what you are feeling today..."):
            with st.chat_message("user"):
                st.write(prompt)

            processed = _process_turn(st.session_state.active_user_id, prompt)
            with st.chat_message("assistant"):
                emo = processed["extraction"]["emotion"]
                risk_lvl = processed["risk"]["level"]
                st.markdown(f"*Detected emotion:* **{emo.upper()}** | *Risk:* **{risk_lvl.upper()}**")

        st.divider()

        st.subheader("Temporal Personal Causal Graph")
        graph_payload = st.session_state.engine.get_user_graph(st.session_state.active_user_id)
        if graph_payload["nodes"]:
            json_path = st.session_state.engine.export_user_graph_json(st.session_state.active_user_id)
            html_path = st.session_state.engine.export_user_xai_html(st.session_state.active_user_id)

            st.download_button(
                label="Download Graph JSON",
                data=Path(json_path).read_bytes(),
                file_name=Path(json_path).name,
                mime="application/json",
            )

            html_content = Path(html_path).read_text(encoding="utf-8")
            components.html(html_content, height=750, scrolling=True)
            st.caption("Visual explanation of underlying triggers, mechanisms, and symptoms.")
        else:
            st.info("No data yet. Start a conversation to build the causal graph.")

    with side_col:
        st.subheader("Current Risk Assesment")
        latest = _latest_for_user(st.session_state.active_user_id)
        if latest:
            risk_score = float(latest["risk"]["score"])
            risk_level = str(latest["risk"]["level"])
            st.metric("Risk Level", risk_level.upper())
            st.metric("Risk Score", f"{risk_score:.2f}")
            st.progress(min(max(risk_score, 0.0), 1.0))
            st.markdown(
                f"<span style='color:{_risk_color(risk_level)}; font-weight:bold;'>Top reasons contributing to risk:</span>",
                unsafe_allow_html=True,
            )
            for reason in latest["risk"].get("reasons", []):
                st.markdown(f"- {reason}")
        else:
            st.info("Start chatting to see risk assessment.")

        st.divider()

        st.subheader("Entity Extraction Summary")
        if latest:
            extraction = latest["extraction"]
            stat1, stat2 = st.columns(2)
            stat1.metric("Emotion", extraction["emotion"].title())
            stat2.metric("Nodes Extracted", latest["graph_stats"]["node_count"])

            st.markdown("**Triggers**")
            for item in extraction["triggers"] or ["*none*"]: st.markdown(f"- {item}")

            st.markdown("**Mechanisms**")
            for item in extraction["mechanisms"] or ["*none*"]: st.markdown(f"- {item}")

            st.markdown("**Symptoms**")
            for item in extraction["symptoms"] or ["*none*"]: st.markdown(f"- {item}")

            st.divider()
            st.subheader("Raw History")
            for item in reversed(history[-5:]):
                t = item["turn"]
                e = item["extraction"]
                with st.expander(f"{t['timestamp']} | Emotion: {e['emotion']}"):
                    st.write("**Text:**", t["text"])
                    st.json({
                        "triggers": e["triggers"],
                        "mechanisms": e["mechanisms"],
                        "symptoms": e["symptoms"]
                    })
        else:
            st.info("Process a turn to view extraction details.")


def main() -> None:
    _init_state()
    _render_dashboard()


if __name__ == "__main__":
    main()
