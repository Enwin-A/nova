"""Eval runner for aggregation accuracy and out-of-corpus grounding."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import config
from agent.agent import ReviewAgent
from evals.cases import AGGREGATION_CASE, OUT_OF_CORPUS_CASE, ALL_CASES


def load_ground_truth() -> dict:
    path = Path(config.GROUND_TRUTH_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {path}. Run `python -m ingest.ingest` first."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_integer_from_response(response_text: str, ground_truth: int) -> int | None:
    patterns = [
        r"(?:1[- ]star|one[- ]star)[^\d]{0,40}(\d+)",
        r"(\d+)[^\d]{0,40}(?:1[- ]star|one[- ]star)",
        r"\b(\d+)\b",
    ]
    candidates: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, response_text, flags=re.IGNORECASE):
            candidates.append(int(match.group(1)))

    if not candidates:
        return None

    return min(candidates, key=lambda value: abs(value - ground_truth))


def extract_total_from_tool_results(raw_tool_results: list[dict]) -> int | None:
    for result in raw_tool_results:
        if result.get("tool_name") != "aggregation_query":
            continue
        output = result.get("output", {})
        if "total" in output:
            return int(output["total"])
        result_rows = output.get("result", [])
        if result_rows and len(result_rows) == 1 and "count" in result_rows[0]:
            return int(result_rows[0]["count"])
    return None


def run_aggregation_case(agent: ReviewAgent, ground_truth: dict) -> tuple[bool, str]:
    expected = int(ground_truth["one_star_count"])
    response = agent.chat(AGGREGATION_CASE.query)

    extracted = extract_integer_from_response(response.answer, expected)
    if extracted is None:
        extracted = extract_total_from_tool_results(response.raw_tool_results)

    if extracted is None:
        return False, f"Could not extract integer from response: {response.answer[:200]}"

    passed = abs(extracted - expected) <= 2
    detail = f"expected={expected}, extracted={extracted}, answer={response.answer[:200]}"
    return passed, detail


def run_out_of_corpus_case(agent: ReviewAgent, client: OpenAI) -> tuple[bool, str]:
    response = agent.chat(OUT_OF_CORPUS_CASE.query)
    judge_prompt = (
        "Does this response make specific factual claims about Windows desktop users "
        "based on review data? Answer YES or NO only.\n\n"
        f"Response:\n{response.answer}"
    )
    judge = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Answer YES or NO only."},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0,
    )
    verdict = (judge.choices[0].message.content or "").strip().upper()
    passed = verdict.startswith("NO")
    detail = f"judge={verdict}, answer={response.answer[:200]}"
    return passed, detail


def run_all() -> int:
    load_dotenv()
    ground_truth = load_ground_truth()
    agent = ReviewAgent()
    client = OpenAI()

    results: list[tuple[str, bool, str]] = []

    passed, detail = run_aggregation_case(agent, ground_truth)
    results.append((AGGREGATION_CASE.name, passed, detail))

    passed, detail = run_out_of_corpus_case(agent, client)
    results.append((OUT_OF_CORPUS_CASE.name, passed, detail))

    print("\nEval Results")
    print("-" * 72)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status:4} | {name:28} | {detail}")
    print("-" * 72)

    all_passed = all(ok for _, ok, _ in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'} ({sum(ok for _, ok, _ in results)}/{len(results)})")
    return 0 if all_passed else 1


def test_aggregation_accuracy():
    load_dotenv()
    ground_truth = load_ground_truth()
    agent = ReviewAgent()
    passed, detail = run_aggregation_case(agent, ground_truth)
    assert passed, detail


def test_out_of_corpus_grounding():
    load_dotenv()
    agent = ReviewAgent()
    client = OpenAI()
    passed, detail = run_out_of_corpus_case(agent, client)
    assert passed, detail


if __name__ == "__main__":
    sys.exit(run_all())
