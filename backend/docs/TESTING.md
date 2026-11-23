# Hebbia AI Backend — Testing Guide

How to test every major feature of this backend. Run all commands from the `backend/` directory unless noted.

```powershell
cd C:\Users\pc\OneDrive\Desktop\hebbia-ai\backend
```

---

## 0. Prerequisites

### Install dependencies

```powershell
uv sync --all-groups
```

Optional for pytest (if not already installed):

```powershell
uv add --dev pytest
```

### Environment

Copy `.env.example` → `.env` and fill real values. Do not commit `.env`.

| Feature area | Required env vars |
|---|---|
| App boot / health | `DATABASE_URL` (required by settings) |
| SQLModel / Alembic | `DATABASE_URL` (Postgres / Supabase) |
| Phase 2 ingest (batch) | AWS keys, `AWS_S3_EMBEDDING_BUCKET`, `BEDROCK_BATCH_ROLE_ARN`, Pinecone |
| Phase 2 ingest (ondemand) | AWS keys, Pinecone; set `BEDROCK_INGEST_MODE=ondemand` |
| Phase 3 research API | AWS keys (Bedrock chat + embeddings), `PINECONE_API_KEY`, index with ingested data |
| Local reranker | First run downloads `RERANKER_MODEL` (e.g. BGE); GPU optional |

Supported tickers: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`, `WMT`, `ORCL`.

---

## 1. Lint and format (quality gate)

```powershell
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
uv run ruff format --check .
```

**Pass:** no remaining lint errors; format check exits 0.

---

## 2. Automated tests

Test files live under `tests/unit/` and `tests/integration/`.

```powershell
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run pytest tests -q
```

| Suite | Covers |
|---|---|
| `tests/unit/ai/test_chunker.py` | Chunking |
| `tests/unit/ai/test_loader_cleaner.py` | Load / clean |
| `tests/unit/ai/test_bedrock_batch.py` | Batch embed helpers |
| `tests/unit/ai/test_bedrock_embeddings.py` | Titan embeddings client |
| `tests/unit/ai/test_bedrock_chat.py` | Bedrock chat factory |
| `tests/unit/ai/test_phase3_schemas.py` | Pydantic research schemas |
| `tests/unit/ai/test_phase3_rag_helpers.py` | RAG helpers |
| `tests/unit/ai/test_query_understanding_heuristic.py` | Query understanding |
| `tests/integration/ai/test_research_api.py` | Health + research SSE (mocked agent) |

**Note:** Some tests may still use older import paths (`ai.*` / `main`). If imports fail, update them to `app.*` (e.g. `from app.main import app`) before relying on CI.

---

## 3. Database (SQLModel + Alembic)

### Migrate

```powershell
uv run alembic upgrade head
uv run alembic current
```

**Pass:** `alembic current` shows the latest revision; Supabase/Postgres has app tables (e.g. `document`) and `alembic_version`.

### Autogenerate after model changes

```powershell
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```

Confirm new revision files import `sqlmodel` if they use SQLModel column types.

---

## 4. HTTP app smoke tests

### Start the server

```powershell
uv run fastapi dev app/main.py
```

Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Health

```powershell
curl http://127.0.0.1:8000/health
```

**Pass:** `{"status":"ok"}` and request logging appears in the terminal (middleware).

### OpenAPI

Visit `/docs` and `/openapi.json` — confirm `GET /health` and `POST /api/v1/research/stream` are listed.

---

## 5. Phase 1 — SEC download and Markdown conversion

These scripts write under `data/downloads/` and `data/markdown/` (gitignored). Do not commit those folders.

### Download filings

```powershell
uv run data/download_sec_filings.py
```

**Pass:**
- Files under `data/downloads/<TICKER>/...`
- `data/downloads/manifest.json` with statuses
- Eight companies configured; date range 2021–2026; 10-K / 10-Q / 8-K (+ amendments)

### Convert to Markdown

```powershell
uv run data/convert_sec_filings_to_markdown.py
```

**Pass:**
- Markdown under `data/markdown/`
- `data/markdown/manifest.json`
- Front matter / conversion statuses present; failures reported, not silent

### Cleanup self-check

```powershell
uv run data/check_markdown_cleanup.py
```

**Pass:** script exits successfully (cleanup helpers behave as expected).

---

## 6. Phase 2 — Ingestion, embeddings, Pinecone

Requires Markdown corpus + AWS (Bedrock) + Pinecone.

### Full ingest + validation

```powershell
uv run python -m app.ai.ingestion
```

### Validation only (existing report / index)

```powershell
uv run python -m app.ai.ingestion --validate-only
```

**Pass:**
- Console prints `Phase 2 validation: PASS`
- Namespaces present for companies
- Retrieval smoke checks OK
- Ingestion report written (see `ingestion_report` path in settings)

**Modes:**
- Default: `BEDROCK_INGEST_MODE=batch` (S3 JSONL + Bedrock batch job)
- Fallback: `BEDROCK_INGEST_MODE=ondemand` (InvokeModel loops)

**Hybrid note:** Dense retrieval always works. Sparse/hybrid BM25 activates only if the Pinecone index metric is `dotproduct` and a BM25 encoder is available (`pinecone-text`). Cosine indexes fall back to dense + RRF.

---

## 7. Phase 3 — Research agent API (SSE)

Endpoint: **`POST /api/v1/research/stream`**

Body schema (`ResearchRequest`):

```json
{
  "question": "What were NVIDIA's major revenue drivers in fiscal 2025?",
  "selected_companies": ["NVDA"],
  "conversation_id": "research-session-001",
  "filing_types": [],
  "reporting_period_start": null,
  "reporting_period_end": null
}
```

### Happy path (live agent)

With the server running and Pinecone ingested:

```powershell
curl -N -X POST http://127.0.0.1:8000/api/v1/research/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What were NVIDIA revenue drivers in fiscal 2025?\",\"selected_companies\":[\"NVDA\"],\"conversation_id\":\"test-001\"}"
```

Or use **Swagger UI** at `/docs` → `POST /api/v1/research/stream`.

**Pass:** SSE stream includes progress events such as:
- `research_started`
- `planning_completed` / retrieval / rerank / calculation-related events (as emitted)
- `answer_completed` (final payload with answer or error fields)
- On failure inside the agent stream: `research_failed`

### Validation / negative cases

| Case | Request | Expected |
|---|---|---|
| Unsupported ticker | `"selected_companies": ["TSLA"]` | **422** validation error |
| Question too short | `"question": "hi"` | **422** (`min_length=3`) |
| Empty companies | `"selected_companies": []` | Agent may mark unsupported / fail gracefully with events |

### Multi-company comparison

```json
{
  "question": "Compare cloud revenue trends for Microsoft and Amazon",
  "selected_companies": ["MSFT", "AMZN"]
}
```

**Pass:** stream completes; answer references both companies or reports insufficient evidence clearly.

---

## 8. Providers (optional isolated checks)

### Bedrock embeddings (query-time)

```powershell
uv run python -c "from app.providers.bedrock_embeddings import BedrockTitanEmbeddings; e=BedrockTitanEmbeddings(); print(len(e.embed_query('test revenue')))"
```

**Pass:** vector length equals `EMBEDDING_DIMENSIONS` (default 1024).

### Bedrock chat

```powershell
uv run python -c "from app.providers.bedrock_chat import get_chat_model; m=get_chat_model(); print(m.invoke('Say ok in one word').content)"
```

**Pass:** non-empty model response (requires Bedrock access to Nova / configured chat model).

### Local reranker

Exercised inside retrieval/rerank during research. First call may download the cross-encoder weights. If the model fails to load, code falls back to retrieval scores (check logs for warnings).

---

## 9. Feature checklist (quick map)

| Feature | How to test |
|---|---|
| Ruff lint/format | §1 |
| Pytest unit/integration | §2 |
| Alembic / Postgres | §3 |
| Health + middleware logging | §4 |
| OpenAPI docs | §4 `/docs` |
| SEC download | §5 download script |
| Markdown conversion | §5 convert + cleanup |
| Bedrock batch/ondemand ingest | §6 |
| Pinecone upsert + namespace validation | §6 |
| Research SSE API | §7 |
| Ticker validation | §7 negative cases |
| Query understanding / tools / generation | §7 live research (end-to-end) |
| Embeddings / chat providers | §8 |

---

## 10. Suggested order for a full manual pass

1. `uv sync --all-groups` + valid `.env`
2. `uv run ruff check .`
3. `uv run alembic upgrade head`
4. `uv run fastapi dev app/main.py` → `/health`
5. Phase 1 scripts (if corpus missing)
6. `uv run python -m app.ai.ingestion`
7. `POST /api/v1/research/stream` for one ticker, then a comparison query
8. `uv run pytest tests -q` (after fixing any stale imports)

---

## 11. What not to push / not to commit while testing

- `.env` and secrets
- `data/downloads/`
- `data/markdown/`
- Local caches / ingestion reports under `data/` (see `.gitignore`)

Keep scripts (`data/*.py`) and this doc in git; keep corpus artifacts local only.
