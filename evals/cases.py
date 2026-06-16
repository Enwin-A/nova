"""Eval case definitions for the review chat agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalCase:
    name: str
    query: str
    description: str


AGGREGATION_CASE = EvalCase(
    name="aggregation_accuracy",
    query="How many 1-star reviews are there?",
    description="Extracted count should be within 2 of ground truth one_star_count",
)

OUT_OF_CORPUS_CASE = EvalCase(
    name="out_of_corpus_grounding",
    query="What do users say about the desktop Windows app?",
    description="Response should not make specific factual claims about Windows desktop users",
)

ALL_CASES = [AGGREGATION_CASE, OUT_OF_CORPUS_CASE]
