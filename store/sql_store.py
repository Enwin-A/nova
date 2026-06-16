"""SQLite store for structured review metadata and aggregation queries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

import config

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    rating INTEGER NOT NULL,
    date TEXT,
    language TEXT,
    country TEXT,
    app_type TEXT,
    author TEXT,
    original_text TEXT NOT NULL,
    topic TEXT,
    sentiment TEXT,
    sarcasm INTEGER NOT NULL DEFAULT 0,
    sarcasm_confidence REAL NOT NULL DEFAULT 0.0,
    source TEXT
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_rating ON reviews(rating);",
    "CREATE INDEX IF NOT EXISTS idx_language ON reviews(language);",
    "CREATE INDEX IF NOT EXISTS idx_country ON reviews(country);",
    "CREATE INDEX IF NOT EXISTS idx_app_type ON reviews(app_type);",
    "CREATE INDEX IF NOT EXISTS idx_topic ON reviews(topic);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment ON reviews(sentiment);",
    "CREATE INDEX IF NOT EXISTS idx_sarcasm ON reviews(sarcasm);",
    "CREATE INDEX IF NOT EXISTS idx_date ON reviews(date);",
]

UPSERT_SQL = """
INSERT OR REPLACE INTO reviews (
    review_id, rating, date, language, country, app_type, author,
    original_text, topic, sentiment, sarcasm, sarcasm_confidence, source
) VALUES (
    :review_id, :rating, :date, :language, :country, :app_type, :author,
    :original_text, :topic, :sentiment, :sarcasm, :sarcasm_confidence, :source
);
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
    if "sarcasm" not in columns:
        conn.execute(
            "ALTER TABLE reviews ADD COLUMN sarcasm INTEGER NOT NULL DEFAULT 0"
        )
    if "sarcasm_confidence" not in columns:
        conn.execute(
            "ALTER TABLE reviews ADD COLUMN sarcasm_confidence REAL NOT NULL DEFAULT 0.0"
        )


def init_db(path: str | None = None) -> sqlite3.Connection:
    db_path = path or config.SQLITE_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_TABLE_SQL)
    _migrate_schema(conn)
    for index_sql in INDEXES:
        conn.execute(index_sql)
    conn.commit()
    return conn


def upsert_review(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(UPSERT_SQL, row)
    conn.commit()


def upsert_reviews(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    conn.executemany(UPSERT_SQL, rows)
    conn.commit()


def get_review(conn: sqlite3.Connection, review_id: str) -> dict[str, Any] | None:
    cursor = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def load_reviews_df(
    conn: sqlite3.Connection | None = None,
    path: str | None = None,
) -> pd.DataFrame:
    if conn is None:
        conn = init_db(path)
    return pd.read_sql_query("SELECT * FROM reviews", conn)


def compute_ground_truth(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    if conn is None:
        conn = init_db()
    df = load_reviews_df(conn)

    reviews_by_language = (
        df["language"].fillna("unknown").value_counts().sort_index().astype(int).to_dict()
    )
    reviews_by_rating = (
        df["rating"].astype(str).value_counts().sort_index().astype(int).to_dict()
    )
    one_star_count = int((df["rating"] == 1).sum())

    return {
        "total_reviews": int(len(df)),
        "one_star_count": one_star_count,
        "reviews_by_language": reviews_by_language,
        "reviews_by_rating": reviews_by_rating,
    }


def write_ground_truth(path: str | None = None, conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    gt_path = path or config.GROUND_TRUTH_PATH
    ground_truth = compute_ground_truth(conn)
    Path(gt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    return ground_truth
