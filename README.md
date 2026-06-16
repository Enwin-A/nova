# Novakid Review Chat Agent

A conversational agent that lets non-technical managers query multilingual App Store reviews in natural language. It supports semantic retrieval (quotes, themes), SQL aggregation (counts, rankings), and cross-country comparison (e.g. Turkish vs Spanish parents on teacher feedback). Answers cite real review IDs and verbatim snippets.

> **Platform note:** Developed and tested on **Windows 11** with **Python 3.10**. Linux/macOS may work but Chroma/PyTorch import order and console encoding fixes were added specifically for Windows.

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

### 6. (Optional) Run evals

```powershell
python -m evals.eval_runner
# or
pytest evals/eval_runner.py -v
```

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
├── evals/                    # Accuracy + grounding evals
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
- **Sarcasm:** Flagged at ingest (`sarcasm`, `sarcasm_confidence`) and filterable in queries.

See [DECISIONS.md](DECISIONS.md) for full architectural rationale.
