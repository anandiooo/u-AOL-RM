from __future__ import annotations

from typing import Dict, List, Sequence


DEFAULT_LEXICONS: Dict[str, List[str]] = {
    "symptoms": [
        "insomnia",
        "sulit tidur",
        "susah tidur",
        "lelah",
        "fatigue",
        "capek terus",
        "hilang fokus",
        "sulit konsentrasi",
        "low mood",
        "mood turun",
    ],
    "triggers": [
        "deadline",
        "ujian",
        "tugas",
        "konflik keluarga",
        "masalah finansial",
        "biaya kuliah",
        "kesepian",
        "tekanan akademik",
    ],
    "mechanisms": [
        "overthinking",
        "ruminasi",
        "menghindar",
        "isolasi diri",
        "begadang",
    ],
}


CANONICAL_MAP: Dict[str, str] = {
    "sulit tidur": "insomnia",
    "susah tidur": "insomnia",
    "capek terus": "lelah",
    "fatigue": "lelah",
    "sulit konsentrasi": "hilang fokus",
    "mood turun": "low mood",
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
