"""Agent tools: semantic search and aggregation queries."""

from __future__ import annotations

from typing import Any

import config
from store import vector_store  # import before sql_store to ensure chromadb DLLs load first
from store import sql_store

VALID_GROUP_BY = {"topic", "language", "country", "app_type", "sentiment", "rating"}
VALID_FILTERS = {
    "rating",
    "language",
    "country",
    "app_type",
    "topic",
    "sentiment",
    "sarcasm",
    "date_start",
    "date_end",
}


def first_n_chars(text: str, n: int = 200) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


def first_n_words(text: str, n: int = 30) -> str:
    words = (text or "").split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n])


def _native(value: Any) -> Any:
    """Convert numpy/pandas scalars to native Python types for JSON serialization."""
    if value is None:
        return None
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _format_search_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in items:
        original_text = item.get("original_text") or ""
        results.append(
            {
                "review_id": item["review_id"],
                "score": item["score"],
                "snippet": first_n_chars(original_text, 200),
                "verbatim_snippet": first_n_words(original_text, 30),
                "rating": _native(item.get("rating")),
                "language": item.get("language"),
                "country": item.get("country"),
                "date": item.get("date"),
                "app_type": item.get("app_type"),
                "topic": item.get("topic"),
                "sentiment": item.get("sentiment"),
                "sarcasm": _native(item.get("sarcasm")),
                "sarcasm_confidence": _native(item.get("sarcasm_confidence")),
            }
        )
    return results


def _run_filtered_search(
    query: str,
    max_results: int,
    language_filter: str | None,
    rating_filter: int | None,
    country_filter: str | None,
    sarcasm_filter: bool | None = None,
) -> list[dict[str, Any]]:
    raw_results = vector_store.query_reviews(
        query=query,
        max_results=max_results,
        language_filter=language_filter,
        rating_filter=rating_filter,
        country_filter=country_filter,
        sarcasm_filter=sarcasm_filter,
    )
    if not raw_results:
        return []

    has_metadata_filter = any(
        [
            language_filter,
            rating_filter is not None,
            country_filter,
            sarcasm_filter is not None,
        ]
    )
    threshold = (
        config.FILTERED_SIMILARITY_THRESHOLD
        if has_metadata_filter
        else config.SIMILARITY_THRESHOLD
    )
    return [item for item in raw_results if item["score"] >= threshold]


def semantic_search(
    query: str,
    max_results: int = 5,
    language_filter: str | None = None,
    rating_filter: int | None = None,
    country_filter: str | None = None,
    sarcasm_filter: bool | None = None,
) -> dict[str, Any]:
    # Prefer combined country + language filters for segment queries, then relax.
    filter_attempts: list[tuple[str | None, str | None, str]] = []
    if country_filter and language_filter:
        filter_attempts.extend(
            [
                (country_filter, language_filter, "country_and_language"),
                (country_filter, None, "country"),
                (None, language_filter, "language"),
            ]
        )
    elif country_filter:
        filter_attempts.append((country_filter, None, "country"))
    elif language_filter:
        filter_attempts.append((None, language_filter, "language"))
    else:
        filter_attempts.append((None, None, "unfiltered"))

    for country, language, scope in filter_attempts:
        above_threshold = _run_filtered_search(
            query=query,
            max_results=max_results,
            language_filter=language,
            rating_filter=rating_filter,
            country_filter=country,
            sarcasm_filter=sarcasm_filter,
        )
        if not above_threshold:
            continue

        payload: dict[str, Any] = {
            "no_result": False,
            "results": _format_search_results(above_threshold),
        }
        if scope not in {"unfiltered", "country_and_language"}:
            payload["filter_scope_used"] = scope
        return payload

    return {"no_result": True, "results": []}


def cross_country_comparison(
    countries: list[str],
    focus: str,
    topic_filter: str | None = None,
    max_quotes_per_country: int | None = None,
) -> dict[str, Any]:
    """Hybrid comparison: per-country aggregation patterns + semantic quotes."""
    max_quotes = max_quotes_per_country or config.COMPARISON_MAX_QUOTES
    country_codes = [str(c).lower() for c in countries if c][: config.COMPARISON_MAX_COUNTRIES]

    if len(country_codes) < 2:
        return {
            "no_result": True,
            "error": "At least two country codes are required (e.g. tr, es).",
            "segments": [],
        }

    segments: list[dict[str, Any]] = []
    for country in country_codes:
        base_filters: dict[str, Any] = {"country": country}
        topic_filters = {**base_filters, "topic": topic_filter} if topic_filter else base_filters

        topic_breakdown = aggregation_query(
            group_by="topic",
            filters=base_filters,
            n=5,
        )
        sentiment_breakdown = aggregation_query(
            group_by="sentiment",
            filters=topic_filters if topic_filter else base_filters,
            n=3,
        )
        quotes = semantic_search(
            query=focus,
            country_filter=country,
            language_filter=country,
            max_results=max_quotes,
        )

        segments.append(
            {
                "country": country,
                "review_count": topic_breakdown.get("total", 0),
                "top_topics": topic_breakdown.get("result", []),
                "sentiment_breakdown": sentiment_breakdown.get("result", []),
                "quotes": quotes.get("results", []),
                "quotes_no_result": quotes.get("no_result", True),
            }
        )

    return {
        "no_result": all(segment["review_count"] == 0 for segment in segments),
        "focus": focus,
        "topic_filter": topic_filter,
        "countries": country_codes,
        "segments": segments,
    }


def aggregation_query(
    group_by: str,
    filters: dict[str, Any] | None = None,
    metric: str = "count",
    n: int = 5,
) -> dict[str, Any]:
    filters = filters or {}
    applied_filters = {k: v for k, v in filters.items() if k in VALID_FILTERS and v is not None}

    if group_by not in VALID_GROUP_BY:
        return {
            "no_result": True,
            "result": [],
            "total": 0,
            "applied_filters": applied_filters,
            "error": f"Invalid group_by '{group_by}'. Valid values: {sorted(VALID_GROUP_BY)}",
        }

    conn = sql_store.init_db()
    df = sql_store.load_reviews_df(conn)

    if "rating" in applied_filters:
        df = df[df["rating"] == int(applied_filters["rating"])]
    if "language" in applied_filters:
        df = df[df["language"] == str(applied_filters["language"])]
    if "country" in applied_filters:
        df = df[df["country"] == str(applied_filters["country"])]
    if "app_type" in applied_filters:
        df = df[df["app_type"] == str(applied_filters["app_type"])]
    if "topic" in applied_filters:
        df = df[df["topic"] == str(applied_filters["topic"])]
    if "sentiment" in applied_filters:
        df = df[df["sentiment"] == str(applied_filters["sentiment"])]
    if "sarcasm" in applied_filters:
        df = df[df["sarcasm"] == int(bool(applied_filters["sarcasm"]))]
    if "date_start" in applied_filters:
        df = df[df["date"] >= str(applied_filters["date_start"])]
    if "date_end" in applied_filters:
        df = df[df["date"] <= str(applied_filters["date_end"])]

    total = int(len(df))
    if total == 0:
        return {
            "no_result": True,
            "result": [],
            "total": 0,
            "applied_filters": applied_filters,
        }

    if metric != "count":
        grouped = df.groupby(group_by).size().reset_index(name="count")
    else:
        grouped = df.groupby(group_by).size().reset_index(name="count")

    grouped = grouped.sort_values("count", ascending=False).head(n)
    result = [
        {"group": _native(row[group_by]), "count": int(row["count"])}
        for _, row in grouped.iterrows()
    ]

    return {
        "no_result": False,
        "result": result,
        "total": total,
        "applied_filters": applied_filters,
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Semantic search over app store reviews for qualitative questions, "
                "examples, quotes, and themes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of reviews to return",
                        "default": 5,
                    },
                    "language_filter": {
                        "type": "string",
                        "description": (
                            "Optional ISO 639-1 language code for detected review text "
                            "(e.g. tr, es, ru). For country-specific segments such as "
                            "'Turkish parents', set this together with country_filter."
                        ),
                    },
                    "rating_filter": {
                        "type": "integer",
                        "description": "Optional star rating filter (1-5)",
                    },
                    "country_filter": {
                        "type": "string",
                        "description": (
                            "Optional ISO 3166-1 alpha-2 App Store country code "
                            "(e.g. tr for Turkey, es for Spain). For country-specific "
                            "segments, combine with language_filter when the language is known."
                        ),
                    },
                    "sarcasm_filter": {
                        "type": "boolean",
                        "description": (
                            "Optional filter for sarcastic/ironic complaint reviews. "
                            "Use true for queries like 'sarcastic app complaints'."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregation_query",
            "description": (
                "Aggregate review statistics for counts, rankings, top-N by segment, "
                "and comparisons."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": sorted(VALID_GROUP_BY),
                        "description": "Field to group results by",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters to apply before aggregation",
                        "properties": {
                            "rating": {"type": "integer"},
                            "language": {"type": "string"},
                            "country": {"type": "string"},
                            "app_type": {"type": "string"},
                            "topic": {"type": "string"},
                            "sentiment": {"type": "string"},
                            "sarcasm": {"type": "boolean"},
                            "date_start": {"type": "string"},
                            "date_end": {"type": "string"},
                        },
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["count"],
                        "default": "count",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of top groups to return",
                        "default": 5,
                    },
                },
                "required": ["group_by"],
            },
        },
    },
]


CROSS_COUNTRY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "cross_country_comparison",
        "description": (
            "Compare two or more App Store countries on a shared theme. "
            "Runs per-country topic/sentiment aggregation plus semantic quote retrieval. "
            "Use first for questions like 'how Turkish and Spanish parents differ on teacher feedback'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ISO country codes to compare, e.g. ['tr', 'es']",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Shared theme to compare, e.g. 'teacher feedback' or 'teacher quality'"
                    ),
                },
                "topic_filter": {
                    "type": "string",
                    "enum": sorted(
                        {
                            "teacher_quality",
                            "app_bugs",
                            "pricing",
                            "content",
                            "progress_tracking",
                            "technical_issues",
                            "other",
                        }
                    ),
                    "description": (
                        "Optional topic filter when the question is about a specific theme "
                        "(e.g. teacher_quality for teacher feedback)."
                    ),
                },
                "max_quotes_per_country": {
                    "type": "integer",
                    "description": "Supporting quotes per country",
                    "default": 3,
                },
            },
            "required": ["countries", "focus"],
        },
    },
}

COMPARISON_TOOL_SCHEMAS = TOOL_SCHEMAS + [CROSS_COUNTRY_TOOL_SCHEMA]


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "semantic_search":
        return semantic_search(**arguments)
    if name == "aggregation_query":
        return aggregation_query(**arguments)
    if name == "cross_country_comparison":
        return cross_country_comparison(**arguments)
    return {"error": f"Unknown tool: {name}"}
