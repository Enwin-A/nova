"""Lightweight query router — heuristics only, no extra LLM call."""

from __future__ import annotations

import re

COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bcontrast\b",
    r"\bdiffer(ence|ences|ent|s)?\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bbetween .+ and .+\b",
    r"\bhow .+ and .+ (compare|differ)\b",
]

COUNTRY_SIGNALS = {
    "turkish": "tr",
    "turkey": "tr",
    "spanish": "es",
    "spain": "es",
    "romanian": "ro",
    "romania": "ro",
    "russian": "ru",
    "russia": "ru",
    "polish": "pl",
    "poland": "pl",
    "german": "de",
    "germany": "de",
    "italian": "it",
    "italy": "it",
    "korean": "ko",
    "korea": "ko",
    "arabic": "ar",
    "american": "us",
    "united states": "us",
}


def _detect_country_codes(message: str) -> list[str]:
    lower = message.lower()
    found: list[str] = []
    seen: set[str] = set()

    for alias, code in COUNTRY_SIGNALS.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower) and code not in seen:
            seen.add(code)
            found.append(code)

    for code in re.findall(r"\b([a-z]{2})\b", lower):
        if code in {"tr", "es", "ro", "ru", "pl", "de", "it", "ko", "us", "ar", "en"}:
            if code not in seen:
                seen.add(code)
                found.append(code)

    return found


def is_comparison_query(message: str) -> bool:
    """Return True for cross-segment comparison questions (no LLM call)."""
    lower = message.lower().strip()
    if not lower:
        return False

    if any(re.search(pattern, lower) for pattern in COMPARISON_PATTERNS):
        return True

    return len(_detect_country_codes(lower)) >= 2


def extract_comparison_countries(message: str) -> list[str]:
    """Best-effort country code extraction for comparison tool calls."""
    return _detect_country_codes(message)
