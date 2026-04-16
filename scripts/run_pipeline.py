from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tcmh_chatbot.chatbot.engine import TemporalCausalChatbot
from tcmh_chatbot.core.schemas import ProcessTurnRequest


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal causal chatbot pipeline on JSONL conversations")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample" / "sample_conversations.jsonl",
        help="Path to JSONL conversation file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    if not records:
        print("No records found.")
        return

    engine = TemporalCausalChatbot()
    users = set()

    for row in records:
        request = ProcessTurnRequest(
            user_id=row["user_id"],
            turn_id=row.get("turn_id"),
            timestamp=row.get("timestamp"),
            text=row["text"],
        )
        result = engine.process_turn(request)
        users.add(result.turn.user_id)
        print(
            f"[{result.turn.timestamp.isoformat()}] user={result.turn.user_id} "
            f"emotion={result.extraction.emotion} risk={result.risk.level} ({result.risk.score})"
        )

    for user_id in sorted(users):
        json_path = engine.export_user_graph_json(user_id)
        html_path = engine.export_user_xai_html(user_id)
        print(f"Exported graph for {user_id}: {json_path}")
        print(f"Exported XAI view for {user_id}: {html_path}")


if __name__ == "__main__":
    main()
