"""Review chat agent with OpenAI tool-calling loop."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from openai import OpenAI

import config
from agent import prompts
from agent.router import is_comparison_query
from agent.tools import COMPARISON_TOOL_SCHEMAS, TOOL_SCHEMAS, execute_tool

logger = structlog.get_logger(__name__)


@dataclass
class Citation:
    review_id: str
    rating: Any
    language: str
    snippet: str


@dataclass
class ChatResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    no_result: bool = False
    mode: str = "simple"
    raw_tool_results: list[dict[str, Any]] = field(default_factory=list)


def _ensure_trace_dir() -> None:
    Path(config.TRACE_PATH).parent.mkdir(parents=True, exist_ok=True)


def _append_trace(entry: dict[str, Any]) -> None:
    _ensure_trace_dir()
    with open(config.TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_review_ids(tool_output: dict[str, Any]) -> list[str]:
    if "results" in tool_output and isinstance(tool_output["results"], list):
        return [r.get("review_id") for r in tool_output["results"] if r.get("review_id")]
    return []


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to native Python JSON types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    return obj


def _configure_stdio_utf8() -> None:
    """Prevent UnicodeEncodeError when logging multilingual text on Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()


def _summarize_tool_output(tool_output: dict[str, Any]) -> str:
    summary = json.dumps(_to_json_safe(tool_output), ensure_ascii=True)
    return summary[:200]


def _collect_citations(tool_results: list[dict[str, Any]]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for result in tool_results:
        output = result.get("output", {})
        if result.get("tool_name") == "cross_country_comparison":
            for segment in output.get("segments", []):
                for item in segment.get("quotes", []):
                    review_id = item.get("review_id")
                    if not review_id or review_id in seen:
                        continue
                    seen.add(review_id)
                    citations.append(
                        Citation(
                            review_id=review_id,
                            rating=item.get("rating"),
                            language=str(item.get("language") or "unknown"),
                            snippet=item.get("verbatim_snippet") or item.get("snippet") or "",
                        )
                    )
            continue
        if result.get("tool_name") != "semantic_search":
            continue
        for item in output.get("results", []):
            review_id = item.get("review_id")
            if not review_id or review_id in seen:
                continue
            seen.add(review_id)
            citations.append(
                Citation(
                    review_id=review_id,
                    rating=item.get("rating"),
                    language=str(item.get("language") or "unknown"),
                    snippet=item.get("verbatim_snippet") or item.get("snippet") or "",
                )
            )
    return citations


def _detect_no_result(tool_results: list[dict[str, Any]]) -> bool:
    if not tool_results:
        return False
    return all(r.get("output", {}).get("no_result") for r in tool_results)


class ReviewAgent:
    def __init__(self) -> None:
        load_dotenv()
        self.client = OpenAI()

    def chat(self, user_message: str, history: list[tuple[str, str]] | None = None) -> ChatResponse:
        history = history or []
        mode = "comparison" if is_comparison_query(user_message) else "simple"
        system_prompt = (
            prompts.COMPARISON_SYSTEM_PROMPT if mode == "comparison" else prompts.SYSTEM_PROMPT
        )
        tool_schemas = COMPARISON_TOOL_SCHEMAS if mode == "comparison" else TOOL_SCHEMAS

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for user_turn, assistant_turn in history:
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": assistant_turn})

        messages.append({"role": "user", "content": user_message})

        tools_used: list[str] = []
        raw_tool_results: list[dict[str, Any]] = []

        for _ in range(config.MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message

            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    start = time.perf_counter()
                    try:
                        arguments = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}

                    output = execute_tool(tool_name, arguments)
                    latency_ms = int((time.perf_counter() - start) * 1000)

                    tools_used.append(tool_name)
                    raw_tool_results.append(
                        {"tool_name": tool_name, "input": arguments, "output": output}
                    )

                    _append_trace(
                        {
                            "timestamp": time.time(),
                            "query": user_message,
                            "mode": mode,
                            "tool_name": tool_name,
                            "tool_input": arguments,
                            "tool_output_summary": _summarize_tool_output(output),
                            "review_ids": _extract_review_ids(output),
                            "latency_ms": latency_ms,
                        }
                    )

                    logger.info(
                        "tool_call",
                        query=user_message,
                        tool_name=tool_name,
                        tool_input=arguments,
                        tool_output_summary=_summarize_tool_output(output),
                        review_ids=_extract_review_ids(output),
                        latency_ms=latency_ms,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(_to_json_safe(output), ensure_ascii=False),
                        }
                    )
                continue

            answer = message.content or prompts.CORPUS_SILENCE_MESSAGE
            citations = _collect_citations(raw_tool_results)
            no_result = _detect_no_result(raw_tool_results)

            return ChatResponse(
                answer=answer,
                citations=citations,
                tools_used=list(dict.fromkeys(tools_used)),
                no_result=no_result,
                mode=mode,
                raw_tool_results=raw_tool_results,
            )

        return ChatResponse(
            answer=(
                "I was unable to complete this request within the allowed number of tool calls. "
                "Please try a simpler question."
            ),
            citations=_collect_citations(raw_tool_results),
            tools_used=list(dict.fromkeys(tools_used)),
            no_result=True,
            mode=mode,
            raw_tool_results=raw_tool_results,
        )
