"""Unit tests for the comparison query router (no API key required)."""

from __future__ import annotations

import pytest

from agent.router import extract_comparison_countries, is_comparison_query


@pytest.mark.parametrize(
    "query,expected",
    [
        ("How do Turkish parents and Spanish parents differ on teacher feedback?", True),
        ("Compare Turkish vs Spanish reviews about teachers", True),
        ("Turkish and Spanish parents — what's the difference?", True),
        ("How does tr compare to es on pricing?", True),
        ("Contrast Romania and Poland on app bugs", True),
        ("How many 1-star reviews are there?", False),
        ("What are users complaining about most in 1-star reviews?", False),
        ("Show me a review where someone praised the teachers", False),
        ("What are Turkish parents saying?", False),
        ("What are Spanish-speaking parents saying?", False),
        ("What do Arabic-speaking parents say about teachers?", False),
        ("", False),
    ],
)
def test_is_comparison_query(query: str, expected: bool) -> None:
    assert is_comparison_query(query) is expected


@pytest.mark.parametrize(
    "query,expected_codes",
    [
        ("Turkish vs Spanish on teachers", ["tr", "es"]),
        ("Compare tr and es reviews", ["tr", "es"]),
        ("Romanian parents feedback", ["ro"]),
        ("How many reviews?", []),
    ],
)
def test_extract_comparison_countries(query: str, expected_codes: list[str]) -> None:
    assert extract_comparison_countries(query) == expected_codes
