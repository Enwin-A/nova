# Novakid Review Chat Agent

A conversational agent that lets non-technical managers query multilingual App Store reviews in natural language. It supports semantic retrieval (quotes, themes), SQL aggregation (counts, rankings), and cross-country comparison (e.g. Turkish vs Spanish parents on teacher feedback). Answers cite real review IDs and verbatim snippets.

> **Platform note:** Developed and tested on **Windows 11** with **Python 3.10 and 3.11**. Linux/macOS may work but Chroma/PyTorch import order and console encoding fixes were added specifically for Windows.

---

## Architecture (5 bullets)

1. **Agent loop** — GPT-4o-mini tool-calling loop (`agent/agent.py`): up to 5 iterations, synthesizes a grounded English answer from tool outputs only.
2. **Retriever (qualitative)** — ChromaDB + BGE-M3 embeddings over original review text; metadata filters for country, language, rating, sarcasm.
3. **Structured store (quantitative)** — SQLite (`data/reviews.db`) with topic, sentiment, country, language, rating; pandas GROUP BY via `aggregation_query`.
4. **Router** — LLM picks `semantic_search` vs `aggregation_query`; a zero-LLM heuristic in `agent/router.py` switches **comparison mode** (exposes `cross_country_comparison`) when the query mentions two countries or compare/differ/vs.
5. **UI** — Gradio chat with example queries, citation footer, and JSONL trace logging to `traces/agent_trace.jsonl`.

See [DECISIONS.md](DECISIONS.md) for deeper rationale on each choice.

---

## Models & multilingual strategy

| Step | Model | Why |
|---|---|---|
| Agent routing + synthesis | GPT-4o-mini | Cheap, fast tool-calling; good enough for manager-facing summaries |
| Ingest topic + sarcasm labels | GPT-4o-mini (JSON mode) | Structured classification over original text |
| Embeddings | BAAI/bge-m3 | Strong multilingual retrieval; embed **original** text with `passage:` / `query:` prefixes |
| Eval grounding judge | GPT-4o-mini | Strict YES/NO on whether the agent hallucinated out-of-corpus claims |

**Multilingual approach:** no translation at ingest. BGE-M3 retrieves across Turkish, Spanish, Arabic, Russian, etc. in the user's original language. At answer time, GPT adds an English translation in parentheses after each non-English quote so managers can read the response.

**What would change this:** if evals showed cross-lingual recall gaps, we'd add persisted `english_text` at ingest (NLLB or GPT batch). If aggregation topic labels were noisy, we'd switch to a smaller fixed taxonomy with human spot-checks.
---

## Eval design

### How evals work

Evals live in `evals/`. Ground truth counts are written automatically during ingest to `evals/eval_ground_truth.json` (e.g. `one_star_count: 151` for the 1,031-review corpus).

**Run all evals:**

```powershell
python -m evals.eval_runner
# Router unit tests only (no API key):
pytest evals/test_router.py -v
# Full suite (router + agent; needs OPENAI_API_KEY + completed ingest):
pytest evals/ -v
```

Each case returns **PASS/FAIL** with a detail string. Agent evals call the real `ReviewAgent` — they are integration tests, not mocks.

### Shipped cases (3)

| Case | Query | What it tests | Grader |
|---|---|---|---|
| `aggregation_accuracy` | "How many 1-star reviews are there?" | Count questions hit SQL, not vibes | Extract integer from answer (or tool `total`); pass if within ±2 of ground truth |
| `aggregation_routing` | "What is the #1 complaint topic in 1-star reviews?" | Top-N / ranking questions route to structured tool | Pass if `aggregation_query` called with `group_by=topic` and `filters.rating=1` |
| `out_of_corpus_grounding` | "What do users say about the desktop Windows app?" | Agent refuses when corpus has nothing relevant | GPT-4o-mini judge: "Does the response make specific factual claims about Windows desktop users?" — pass if **NO** |

**Case 1 flow:** `agent.chat()` → LLM should call `aggregation_query` → grader parses "151" from the answer or reads `total` from raw tool output → compares to `eval_ground_truth.json`.

**Case 2 flow:** same agent call → grader inspects `response.raw_tool_results` for the correct tool name and arguments (testable routing without parsing natural language).

**Case 3 flow:** semantic search returns `no_result: true` (similarity below threshold) → grader checks the final answer does not invent Windows desktop feedback. Early traces showed a failure mode where generic "Great app" hits passed the old threshold; raising `SIMILARITY_THRESHOLD` to 0.65 fixed that.

### Cases we'd add next

| Future case | Grader |
|---|---|
| Multilingual recall | Fixed query in English ("teacher quality"); pass if ≥1 returned review has `language != en` and snippet mentions teachers (keyword or LLM judge) |
| Hybrid query | Canonical brief query (top 1-star complaints + 3 quotes + Spanish segment); pass if both `aggregation_query` and `semantic_search` called |
| Cross-country comparison | "Turkish vs Spanish on teachers"; pass if `cross_country_comparison` invoked with `countries=['tr','es']` |
| Date filter | "1-star reviews this year"; pass if `aggregation_query` filters include plausible `date_start`/`date_end` |
| Citation integrity | Random sample of cited `review_id`s; pass if ID exists in SQLite and snippet is substring of `original_text` |

---

## Failure modes observed

1. **Segment queries returning empty (Turkish parents)** — Early runs used `language_filter="tr"` alone; cross-lingual similarity scores were below threshold. **Fix applied:** combined `country_filter` + `language_filter` with fallback chain (country+language → country → language) and lower threshold (0.45) when metadata filters are set. **Next step:** add a multilingual recall eval and tune thresholds per segment.

2. **Out-of-corpus false positives** — Unfiltered semantic search once returned generic 5-star "Great app" reviews for "Windows desktop app" (score ~0.62). **Fix applied:** raised unfiltered threshold to 0.65; prompt instructs corpus-silence when snippets don't mention the asked topic. **Next step:** hard-fail synthesis when all tool results have `no_result: true`.

---

## Next highest-leverage build

A **composite tool** `top_complaints_with_quotes(rating, n_quotes, segment_filters)` that runs aggregation then semantic search in one call. This would make the brief's canonical hybrid query reliable without depending on the LLM calling two tools in the right order.

---

## Trustpilot data sourcing

We would **not scrape Trustpilot** for production. Trustpilot's ToS and robots.txt restrict automated collection, and scraped data is fragile (HTML/JSON layout changes, incomplete history, legal exposure).

**Recommended approach:** request an official export or API access from Novakid's Trustpilot business account. Normalize to the same schema as App Store reviews (`review_id`, `rating`, `body`, `date`, `country` if available, `source="trustpilot"`), run the existing ingest pipeline, and tag `source` so the agent can filter or compare channels.

**Constraints to flag:** different rating semantics vs App Store, possible duplicate users across platforms, sparser locale metadata, and need for deduplication if the same parent reviewed on both stores.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | 3.10 or 3.11 recommended |
| **OpenAI API key** | Used for agent, ingest classification, and evals |
| **~2 GB disk** | BGE-M3 embedding model downloads on first ingest |
| **Network** | OpenAI API + HuggingFace model download during ingest |

All Python dependencies are installed via `requirements.txt` (Gradio, ChromaDB, PyTorch, sentence-transformers, etc.) — nothing else to download manually except your review corpus.

---

## Quick start

### 1. Clone and install

```powershell
cd nova
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API key

Copy the example env file and add your key:

**Windows**

```powershell
copy .env.example .env
```

**macOS/Linux**

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
```

`.env` is gitignored — never commit it.

### 3. Add review data

Place your App Store export here (one JSON object per line):

```
data/reviews.jsonl
```

Expected fields per line: `review_id`, `title`, `body`, `country`, `rating`, `author`, `updated` (and optionally `app`). See an existing line in the corpus for the exact shape. This file is **not** in the repo — you must supply it.

### 4. Run ingest (one-time, ~5–15 min)

```powershell
python -m ingest.ingest
```

This will:

- Normalize reviews → `data/reviews.db` (SQLite)
- Classify topics + sarcasm via GPT → stored in SQLite
- Embed with BGE-M3 → `data/chroma/` (ChromaDB)
- Write eval ground truth → `evals/eval_ground_truth.json`

**Resume / partial runs:**

```powershell
# Re-classify topics/sarcasm only (no re-embedding)
python -m ingest.ingest --reclassify-only

# Re-embed only (if embedding failed mid-run)
python -m ingest.ingest --embed-only
```

### 5. Start the chat UI

```powershell
python -m ui.app
```

Open **http://127.0.0.1:7860** (port configurable via `GRADIO_SERVER_PORT` in the environment).

- **Enter** — send message  
- **Shift+Enter** — new line in the input box  

### 6. Run evals

```powershell
python -m evals.eval_runner
# Router tests only (no API key):
pytest evals/test_router.py -v
# Full suite (needs OPENAI_API_KEY + ingest):
pytest evals/ -v
```

See [Eval design](#eval-design) above for what each case tests.

---

## What you provide vs what gets generated

| Path | You provide? | Description |
|---|---|---|
| `data/reviews.jsonl` | **Yes** | Raw App Store reviews (JSONL) |
| `.env` | **Yes** | `OPENAI_API_KEY` (from `.env.example`) |
| `data/reviews.db` | No | Created by ingest |
| `data/chroma/` | No | Vector index created by ingest |
| `evals/eval_ground_truth.json` | No | Eval reference counts from ingest |
| `traces/agent_trace.jsonl` | No | Runtime tool-call logs from the UI |

Keep `data/.gitkeep` and `traces/.gitkeep` so empty folders exist in a fresh clone.

---

## Requirements (`requirements.txt`)

```
gradio==4.44.1          # Chat UI (port 7860)
chromadb>=0.5,<1.6      # Vector store
pandas>=2.0
sentence-transformers   # BGE-M3 embeddings (pinned for PyTorch 3.10 compat)
transformers            # HuggingFace — pulled by sentence-transformers
torch>=2.0              # CPU embeddings
openai>=1.30            # GPT-4o-mini agent + ingest classification
structlog               # Tool-call tracing
langdetect              # Language detection at ingest
python-dotenv           # Load .env
pytest>=8.0             # Eval runner
```

First ingest downloads **BAAI/bge-m3** (~2 GB) into your HuggingFace cache (user home, not the repo).

---

## Example queries

- What are users complaining about most in 1-star reviews?
- Show me a review where someone praised the teachers
- What are Turkish parents saying?
- How do Turkish parents and Spanish parents differ on teacher feedback?
---

## Project structure

```
nova/
├── config.py                 # Paths, thresholds, model names
├── requirements.txt
├── .env.example              # Template for OPENAI_API_KEY
├── ingest/                   # JSONL → SQLite + Chroma + classification
├── agent/                    # Tool-calling loop + query router
│   ├── router.py             # Simple vs cross-country comparison routing
│   └── tools.py              # semantic_search, aggregation_query, cross_country_comparison
├── store/                    # SQLite + ChromaDB wrappers
├── ui/app.py                 # Gradio chat (monochrome UI)
├── evals/                    # Accuracy, routing, and grounding evals
│   ├── eval_runner.py        # Agent integration evals (3 cases)
│   ├── test_router.py        # Comparison router unit tests (no API)
│   └── cases.py              # Eval query definitions
├── data/
│   ├── .gitkeep
│   └── reviews.jsonl         # ← YOU PLACE THIS FILE HERE
├── traces/                   # agent_trace.jsonl (runtime, gitignored)
├── DECISIONS.md              # Architecture choices
└── README.md
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Ingest fails mid-embedding | `python -m ingest.ingest --embed-only` |
| Port 7860 in use | Stop other Gradio process or set `GRADIO_SERVER_PORT=7861` |
| UI error | Check `traces/agent_trace.jsonl` |
| TensorFlow/Keras crash on import | Already handled in `config.py` (`USE_TF=0`) |
| Windows console Unicode errors | UTF-8 fixes in `agent/agent.py` and `ui/app.py` |

Re-running full ingest is safe and idempotent (`INSERT OR REPLACE` / Chroma upsert).

---

## Translation & sarcasm

- **Translation:** Not stored at ingest. GPT adds English translations in parentheses at answer time for non-English quotes.
- **Sarcasm:** Flagged at ingest (`sarcasm`, `sarcasm_confidence`) and filterable in queries. Quality is imperfect — negative reviews are sometimes over-flagged; see DECISIONS.md.

See [DECISIONS.md](DECISIONS.md) for full architectural rationale and build-steering notes (required submission artifact alongside this README).
