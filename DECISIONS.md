# DECISIONS.md

Architectural decisions for the Novakid Review Chat Agent.

## Router: LLM tool-calling (not a separate classifier)

The agent uses GPT-4o-mini to decide between `semantic_search` and `aggregation_query`. This handles hybrid questions naturally (e.g. "top complaints in 1-star reviews plus three quotes") without a brittle intent classifier.

## Query routing: simple vs cross-country comparison

Before the tool loop, a **heuristic router** (`agent/router.py`) classifies the user message — no extra LLM call.

| Mode | Trigger (examples) | Tools exposed | System prompt |
|---|---|---|---|
| **simple** | "How many 1-star reviews?", "Turkish parents saying" | `semantic_search`, `aggregation_query` | `SYSTEM_PROMPT` |
| **comparison** | "compare", "differ", "Turkish vs Spanish", two+ country signals | above + `cross_country_comparison` | `COMPARISON_SYSTEM_PROMPT` |

**Cross-country flow:** the comparison tool runs per country — topic aggregation, sentiment aggregation, semantic quotes for the shared focus — in one call. The LLM then synthesizes Summary → By country → Key differences with citations. Simple queries never see the comparison tool (fewer tokens, same behaviour as before).

## No translation at ingest

**Override of original spec:** the original spec required `deep-translator` at ingest. We intentionally do not translate reviews at ingest time.

- Reviews are stored as `original_text` only
- BGE-M3 multilingual embeddings are computed on original text
- `TRANSLATION_ENABLED` in `config.py` remains `False` (no Google/deep-translator pipeline)

**Where translation happens:** at **agent response synthesis** only. The system prompt instructs GPT-4o-mini to append an English translation in parentheses after every non-English verbatim quote (`language != "en"`). This is consistent for Spanish, Turkish, Arabic, Russian, etc. Translations are not persisted — they are generated on each chat response. 
NOTE: Can be turned into a persisted field in the future updates.

**Why no ingest translation:** external translation is rate-limited, non-reproducible, and duplicates what GPT can do at answer time for a user-facing UI.

## Topic + sarcasm classification on original text

During ingest, GPT-4o-mini JSON mode classifies each review batch with:

- `topic` — one of the valid topic labels
- `sarcasm` — boolean: ironic/sarcastic **complaint** tone only (not descriptive praise like "the teacher was sarcastic and funny")
- `sarcasm_confidence` — float 0.0–1.0

`sentiment` remains derived from star rating (not GPT) for reproducible SQL filters.

Valid topics: `teacher_quality`, `app_bugs`, `pricing`, `content`, `progress_tracking`, `technical_issues`, `other`.

Stored in SQLite and Chroma metadata. Query via `aggregation_query` filters (`sarcasm=true`) or `semantic_search` (`sarcasm_filter=true`).
NOTE: Upon tested understood the sarcasm detector isnt robust enough and flags negative reviews as sarcasm, it can be fixed in future updates.

## Sentiment from rating (not GPT)

Sentiment is derived deterministically from star rating:

- `rating >= 4` → positive
- `rating <= 2` → negative
- else → neutral

This keeps SQL filters like `sentiment='negative'` aligned with star ratings and reproducible across re-runs.

## LLM

A single `OPENAI_API_KEY` powers:

- Agent router and synthesis (GPT-4o-mini)
- Ingest topic + sarcasm classification (GPT-4o-mini)
- Eval Case 2 LLM judge (GPT-4o-mini)

NOTE: Any bare LLM beyond a certain benchmark would do 

## Embedding model: BGE-M3 over mpnet

**Override of original spec:** use `BAAI/bge-m3` instead of `paraphrase-multilingual-mpnet-base-v2`.

BGE-M3 has stronger multilingual retrieval benchmarks. Documents use the `passage:` prefix and queries use the `query:` prefix per model convention.

## Similarity thresholds

- **Unfiltered semantic search:** `SIMILARITY_THRESHOLD = 0.65` — hallucination guard for out-of-corpus queries (e.g. Windows desktop app).
- **Filtered search** (country, language, rating, or sarcasm filter set): `FILTERED_SIMILARITY_THRESHOLD = 0.45` — segment queries like "Turkish parents" often score lower cross-lingually but are still valid when metadata filters narrow the corpus.

Segment queries use country + language filters together when aligned (e.g. `country_filter="tr"` + `language_filter="tr"`), with fallback to country-only then language-only if the strict match returns nothing.

## UI: Gradio

The implementation uses Gradio on port 7860 (`UI_PORT` / `GRADIO_SERVER_PORT`).

Gradio 4.44 + gradio-client 1.3 has a JSON-schema bug with the default Chatbot "messages" format; the UI uses `type="tuples"` and a small gradio-client schema patch in `ui/app.py`.

## JSONL ingest format

`data/reviews.jsonl` uses App Store fields: `review_id`, `title`, `body`, `country`, `rating`, `updated`, etc. `normalize_review()` combines `title` + `body` into `original_text` and maps `updated` → `date`.

## Citation discipline

The system prompt requires verbatim snippets (≤30 words) from tool results with review_id citations. Snippets come from `original_text` in the review's source language. English translations appear in the agent answer only, in parentheses after the quote.

## Idempotent ingest

SQLite uses `INSERT OR REPLACE` on `review_id`. Chroma uses `upsert` with the same IDs. Re-running ingest is safe.

**Ingest commands:**

| Command | What it does |
|---|---|
| `python -m ingest.ingest` | Full pipeline: normalize → classify → SQLite → embed → Chroma |
| `python -m ingest.ingest --reclassify-only` | Re-classify topic/sarcasm from JSONL, update SQLite, sync Chroma metadata (no re-embed) |
| `python -m ingest.ingest --embed-only` | Re-embed existing SQLite rows into Chroma (resume after embedding failure) |

## What was cut and why

| Cut | Reason |
|---|---|
| Google Play reviews | Out of scope for v1 App Store corpus |
| Ingest-time translation pipeline | Replaced by multilingual embeddings + synthesis-time translation |

## Windows notes

- Chroma must import before pandas in `agent/tools.py` to avoid DLL segfaults
- Console logging uses UTF-8 reconfiguration to handle multilingual trace output
- BGE-M3 runs on CPU (`device="cpu"`) for stability

## Trustpilot reviews:
- URL: https://se.trustpilot.com/_next/data/businessunitprofile-consumersite-2.6966.0/review/novakidschool.com.json?     businessUnit=novakidschool.com&languages=all
 or 
- URL: https://www.trustpilot.com/_next/data/businessunitprofile-consumersite-2.6966.0/review/novakidschool.com.json?page=3&businessUnit=novakidschool.com
Request Method
GET

as examples we can scrape it although it falls under a morally grey area, but as its our own reviews and ratings it wont be that bad, as long as we set our own rate limits to be under the trustpilot's default ratelimits, we should be good, although trustpilots robots.txt file says to avoid reviews and ratings and tries to stop most crawlers etc from scraping it but again its a moral dilema not a technological one, if the need arises we can do it.