"""One-time ingest pipeline: JSONL -> SQLite + ChromaDB + ground truth."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import config
from ingest.enrichment import (
    apply_classifications,
    classify_reviews_batch,
    normalize_review,
)
from store import vector_store  # chromadb DLLs must load before pandas
from store import sql_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_reviews_jsonl(path: str) -> list[dict]:
    reviews_path = Path(path)
    if not reviews_path.exists():
        logger.error("Missing reviews file: %s", path)
        sys.exit(1)

    reviews: list[dict] = []
    with reviews_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON on line %s: %s", line_num, exc)
    return reviews


def build_chroma_metadata(review: dict) -> dict:
    return {
        "review_id": review["review_id"],
        "rating": int(review["rating"]),
        "language": str(review["language"]),
        "country": str(review["country"]),
        "app_type": str(review["app_type"]),
        "topic": str(review["topic"]),
        "sentiment": str(review["sentiment"]),
        "sarcasm": int(review.get("sarcasm") or 0),
        "sarcasm_confidence": float(review.get("sarcasm_confidence") or 0.0),
        "date": str(review["date"]),
    }


def _classify_and_persist(conn, normalized: list[dict], client: OpenAI) -> None:
    logger.info("Classifying topics and sarcasm with %s", config.LLM_MODEL)
    for start in range(0, len(normalized), config.TOPIC_BATCH_SIZE):
        batch = normalized[start : start + config.TOPIC_BATCH_SIZE]
        classifications = classify_reviews_batch(client, batch)
        apply_classifications(batch, classifications)
        logger.info(
            "Classified reviews %s/%s",
            min(start + len(batch), len(normalized)),
            len(normalized),
        )

    sql_store.upsert_reviews(conn, normalized)
    logger.info("Wrote %s reviews to SQLite", len(normalized))


def _sync_chroma_metadata(normalized: list[dict]) -> None:
    logger.info("Syncing Chroma metadata (no re-embedding)")
    for start in range(0, len(normalized), config.EMBED_BATCH_SIZE):
        batch = normalized[start : start + config.EMBED_BATCH_SIZE]
        ids = [r["review_id"] for r in batch]
        metadatas = [build_chroma_metadata(r) for r in batch]
        vector_store.update_review_metadata(ids, metadatas)
        logger.info(
            "Synced metadata %s/%s",
            min(start + len(batch), len(normalized)),
            len(normalized),
        )


def _embed_reviews(normalized: list[dict]) -> None:
    logger.info("Embedding reviews with %s", config.EMBEDDING_MODEL)
    Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    for start in range(0, len(normalized), config.EMBED_BATCH_SIZE):
        batch = normalized[start : start + config.EMBED_BATCH_SIZE]
        ids = [r["review_id"] for r in batch]
        documents = [r["original_text"] for r in batch]
        metadatas = [build_chroma_metadata(r) for r in batch]
        embeddings = vector_store.embed_passages(documents)
        vector_store.upsert_reviews(ids, documents, metadatas, embeddings)
        logger.info("Embedded %s/%s", min(start + len(batch), len(normalized)), len(normalized))


def run_ingest(embed_only: bool = False, reclassify_only: bool = False) -> None:
    load_dotenv()
    conn = sql_store.init_db()

    if embed_only:
        df = sql_store.load_reviews_df(conn)
        if df.empty:
            logger.error("SQLite is empty. Run full ingest first (without --embed-only).")
            sys.exit(1)
        normalized = df.to_dict("records")
        logger.info("Loaded %s reviews from SQLite (embed-only mode)", len(normalized))
        _embed_reviews(normalized)
    elif reclassify_only:
        client = OpenAI()
        raw_reviews = load_reviews_jsonl(config.REVIEWS_JSONL)
        normalized = [normalize_review(raw) for raw in raw_reviews]
        logger.info("Normalized %s reviews (reclassify-only mode)", len(normalized))
        _classify_and_persist(conn, normalized, client)
        _sync_chroma_metadata(normalized)
    else:
        client = OpenAI()
        logger.info("Loading reviews from %s", config.REVIEWS_JSONL)
        raw_reviews = load_reviews_jsonl(config.REVIEWS_JSONL)
        if not raw_reviews:
            logger.error("No reviews found in %s", config.REVIEWS_JSONL)
            sys.exit(1)

        normalized = [normalize_review(raw) for raw in raw_reviews]
        logger.info("Normalized %s reviews", len(normalized))
        _classify_and_persist(conn, normalized, client)
        _embed_reviews(normalized)

    ground_truth = sql_store.write_ground_truth(conn=conn)
    logger.info("Wrote ground truth to %s", config.GROUND_TRUTH_PATH)
    logger.info(
        "Ingest complete: total=%s one_star=%s",
        ground_truth["total_reviews"],
        ground_truth["one_star_count"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest reviews into SQLite and ChromaDB")
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Skip JSONL load and classification; embed existing SQLite rows only",
    )
    parser.add_argument(
        "--reclassify-only",
        action="store_true",
        help=(
            "Re-run GPT topic/sarcasm classification from JSONL, update SQLite, "
            "and sync Chroma metadata without re-embedding"
        ),
    )
    args = parser.parse_args()
    if args.embed_only and args.reclassify_only:
        parser.error("Use only one of --embed-only or --reclassify-only")
    run_ingest(embed_only=args.embed_only, reclassify_only=args.reclassify_only)
