from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


DEFAULT_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "stressed": ["stres", "stressed", "tertekan", "deadline", "kewalahan"],
    "anxious": ["cemas", "anxious", "panik", "khawatir", "takut"],
    "sad": ["sedih", "down", "murung", "kecewa"],
    "hopeless": ["putus asa", "hopeless", "tidak sanggup"],
    "positive": ["lega", "membaik", "tenang", "semangat"],
}


class EmotionDetector:
    """Simple baseline detector to keep the template runnable without heavy models."""

    def __init__(self, emotion_keywords: Dict[str, Sequence[str]] | None = None) -> None:
        self.emotion_keywords: Dict[str, List[str]] = {
            emotion: list(keywords) for emotion, keywords in DEFAULT_EMOTION_KEYWORDS.items()
        }

        if emotion_keywords:
            for emotion, keywords in emotion_keywords.items():
                self.emotion_keywords[emotion] = list(keywords)

    def detect(self, text: str) -> Tuple[str, float, Dict[str, List[str]]]:
        lowered = text.lower()
        hit_count: Dict[str, int] = {}
        matched_keywords: Dict[str, List[str]] = {}

        for emotion, keywords in self.emotion_keywords.items():
            matched = [keyword for keyword in keywords if keyword and keyword in lowered]
            if matched:
                hit_count[emotion] = len(matched)
                matched_keywords[emotion] = matched

        if not hit_count:
            return "neutral", 0.45, {"neutral": []}

        top_emotion = max(hit_count, key=hit_count.get)
        total_hits = sum(hit_count.values())
        confidence = 0.5 + (hit_count[top_emotion] / max(total_hits, 1)) * 0.5

        return top_emotion, round(min(confidence, 0.99), 3), {top_emotion: matched_keywords[top_emotion]}
