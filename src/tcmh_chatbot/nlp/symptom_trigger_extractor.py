from __future__ import annotations

from typing import Dict, List, Sequence


DEFAULT_LEXICONS: Dict[str, List[str]] = {
    "symptoms": [
        "depression",
        "depressed",
        "useless",
        "stay in bed",
    ],
    "triggers": [
        "deadline",
        "boss",
        "project",
    ],
    "mechanisms": [
        "insomnia",
        "3 am",
        "can't shut brain off",
    ],
}


CANONICAL_MAP: Dict[str, str] = {
    "depressed": "depression",
    "can't shut my brain off": "overthinking"
}


class SymptomTriggerExtractor:
    def __init__(self, lexicons: Dict[str, Sequence[str]] | None = None) -> None:
        merged = {bucket: list(values) for bucket, values in DEFAULT_LEXICONS.items()}

        if lexicons:
            for bucket in ("symptoms", "triggers", "mechanisms"):
                if bucket in lexicons:
                    merged[bucket] = list(lexicons[bucket])

        self.lexicons = merged

    @staticmethod
    def _match_terms(text: str, terms: Sequence[str]) -> List[str]:
        matched = [term for term in terms if term and term in text]
        deduplicated = sorted(set(matched), key=lambda item: (len(item), item))
        return deduplicated

    @staticmethod
    def _canonicalize(terms: Sequence[str]) -> List[str]:
        canonical = [CANONICAL_MAP.get(term, term) for term in terms]
        return sorted(set(canonical))

    def extract(self, text: str) -> Dict[str, List[str]]:
        lowered = text.lower()

        symptoms_raw = self._match_terms(lowered, self.lexicons["symptoms"])
        triggers_raw = self._match_terms(lowered, self.lexicons["triggers"])
        mechanisms_raw = self._match_terms(lowered, self.lexicons["mechanisms"])

        symptoms = self._canonicalize(symptoms_raw)
        triggers = self._canonicalize(triggers_raw)
        mechanisms = self._canonicalize(mechanisms_raw)

        return {
            "symptoms": symptoms,
            "triggers": triggers,
            "mechanisms": mechanisms,
            "evidence": {
                "symptoms": symptoms_raw,
                "triggers": triggers_raw,
                "mechanisms": mechanisms_raw,
            },
        }
