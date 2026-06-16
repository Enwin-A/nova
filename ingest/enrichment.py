"""Review enrichment: language detection, sentiment, and GPT metadata classification."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import langdetect
from openai import OpenAI

import config

logger = logging.getLogger(__name__)

REVIEW_CLASSIFY_PROMPT = """Classify each app store review.

Valid topics: {topics}

For each review return:
- index (int): position in the list below
- topic (string): exactly one valid topic
- sarcasm (boolean): true ONLY when sarcasm or irony is used to express a complaint or negative sentiment (e.g. "Great update, now the app crashes every lesson"). false for sincere praise, neutral statements, or non-complaint uses of sarcasm (e.g. "the teacher was sarcastic and funny" as a compliment).
- sarcasm_confidence (float): 0.0 to 1.0 confidence in the sarcasm label

Return JSON with key "results" containing a list of objects, one per review in order.

Reviews:
{reviews_block}
"""


@dataclass
class ReviewClassification:
    topic: str
    sarcasm: bool
    sarcasm_confidence: float


def make_review_id(author: str, date: str, review_text: str) -> str:
    raw = f"{author}|{date}|{review_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def detect_language(text: str) -> str:
    try:
        detected = langdetect.detect(text)
        return detected or "unknown"
    except langdetect.LangDetectException:
        return "unknown"


def derive_sentiment(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def _build_review_text(raw: dict[str, Any]) -> str:
    if raw.get("review_text"):
        return str(raw["review_text"]).strip()
    if raw.get("original_text"):
        return str(raw["original_text"]).strip()
    title = str(raw.get("title") or "").strip()
    body = str(raw.get("body") or "").strip()
    if title and body and title != body:
        return f"{title}\n{body}"
    return title or body


def normalize_review(raw: dict[str, Any]) -> dict[str, Any]:
    review_text = _build_review_text(raw)
    author = str(raw.get("author") or "anonymous")
    date = str(raw.get("date") or raw.get("updated") or "")
    language = raw.get("language") or ""
    if not language or str(language).strip() == "":
        language = detect_language(review_text)

    rating = int(raw.get("rating", 3))
    review_id = str(raw.get("review_id") or make_review_id(author, date, review_text))
    app = str(raw.get("app") or raw.get("app_type") or "main")
    app_type = "main" if "main" in app else app

    return {
        "review_id": review_id,
        "rating": rating,
        "date": date,
        "language": str(language),
        "country": str(raw.get("country") or "unknown"),
        "app_type": app_type,
        "author": author,
        "original_text": review_text,
        "topic": str(raw.get("topic") or "other"),
        "sentiment": derive_sentiment(rating),
        "sarcasm": int(raw.get("sarcasm") or 0),
        "sarcasm_confidence": float(raw.get("sarcasm_confidence") or 0.0),
        "source": str(raw.get("source") or "app_store"),
    }


def _default_classifications(count: int) -> list[ReviewClassification]:
    return [ReviewClassification(topic="other", sarcasm=False, sarcasm_confidence=0.0)] * count


def classify_reviews_batch(
    client: OpenAI,
    reviews: list[dict[str, Any]],
    max_retries: int = 3,
) -> list[ReviewClassification]:
    if not reviews:
        return []

    reviews_block = "\n".join(
        f'{idx}. [{review["review_id"]}] rating={review["rating"]} '
        f'{review["original_text"][:500]}'
        for idx, review in enumerate(reviews)
    )
    prompt = REVIEW_CLASSIFY_PROMPT.format(
        topics=", ".join(config.TOPICS),
        reviews_block=reviews_block,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You classify app store reviews by topic and sarcasm. "
                            "Respond with valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            results = payload.get("results", payload if isinstance(payload, list) else [])
            classifications = _default_classifications(len(reviews))
            for item in results:
                idx = item.get("index")
                if not isinstance(idx, int) or not (0 <= idx < len(reviews)):
                    continue
                topic = item.get("topic", "other")
                if topic not in config.TOPICS:
                    topic = "other"
                sarcasm = bool(item.get("sarcasm", False))
                try:
                    confidence = float(item.get("sarcasm_confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                classifications[idx] = ReviewClassification(
                    topic=topic,
                    sarcasm=sarcasm,
                    sarcasm_confidence=confidence,
                )
            return classifications
        except Exception as exc:
            logger.warning(
                "review_classification_retry",
                extra={"attempt": attempt + 1, "error": str(exc), "batch_size": len(reviews)},
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error("review_classification_failed", extra={"batch_size": len(reviews)})
                return _default_classifications(len(reviews))

    return _default_classifications(len(reviews))


def apply_classifications(
    reviews: list[dict[str, Any]],
    classifications: list[ReviewClassification],
) -> None:
    for review, classification in zip(reviews, classifications):
        review["topic"] = classification.topic
        review["sarcasm"] = int(classification.sarcasm)
        review["sarcasm_confidence"] = classification.sarcasm_confidence


# Backward-compatible alias
def classify_topics_batch(
    client: OpenAI,
    reviews: list[dict[str, Any]],
    max_retries: int = 3,
) -> list[str]:
    classifications = classify_reviews_batch(client, reviews, max_retries=max_retries)
    return [c.topic for c in classifications]
