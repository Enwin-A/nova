"""Central configuration for the Novakid Review Chat Agent."""

import os

# Must be set before any transformers / sentence-transformers import to prevent
# TensorFlow / Keras 3 loading crashes on systems that have TF installed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "gpt-4o-mini"

TRANSLATION_ENABLED = False
# When True in future: run deep-translator/NLLB at ingest and store english_text column.
# Currently all manager-facing translation happens at agent synthesis time (see prompts.py).

SIMILARITY_THRESHOLD = 0.65
# Lower bar when metadata filters already narrow the corpus (e.g. country=tr).
FILTERED_SIMILARITY_THRESHOLD = 0.45
TOP_K_RESULTS = 5
MAX_TOOL_ITERATIONS = 5
COMPARISON_MAX_COUNTRIES = 3
COMPARISON_MAX_QUOTES = 3
EMBED_BATCH_SIZE = 32
TOPIC_BATCH_SIZE = 25

CHROMA_PATH = str(ROOT_DIR / "data" / "chroma")
SQLITE_PATH = str(ROOT_DIR / "data" / "reviews.db")
REVIEWS_JSONL = str(ROOT_DIR / "data" / "reviews.jsonl")
GROUND_TRUTH_PATH = str(ROOT_DIR / "evals" / "eval_ground_truth.json")
TRACE_PATH = str(ROOT_DIR / "traces" / "agent_trace.jsonl")
UI_HOST = "127.0.0.1"
UI_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

TOPICS = [
    "teacher_quality",
    "app_bugs",
    "pricing",
    "content",
    "progress_tracking",
    "technical_issues",
    "other",
]
