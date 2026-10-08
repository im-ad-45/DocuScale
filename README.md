# DocuScale

> **A local-first hybrid RAG engine: BM25 + dense retrieval, cross-encoder reranking, optional HyDE query expansion, and citation-audited answer generation — served through a FastAPI service with Server-Sent Events streaming, and built without orchestration frameworks.**

**Phase 4 Complete:** Client → FastAPI → [HyDE] → BM25 ∥ Dense → RRF → Cross-Encoder Reranking → Grounded Synthesis → Citation Audit → JSON response or SSE stream

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed%20%7C%20ONNX-0A7EA4)](https://github.com/qdrant/fastembed)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-FF4F64)](https://qdrant.tech/)
[![Pydantic](https://img.shields.io/badge/Config-Pydantic%20v2-E92063)](https://docs.pydantic.dev/)
[![Cross-Encoder](https://img.shields.io/badge/Reranker-Cross--Encoder%20%7C%20FastEmbed-2E8B57)](https://github.com/qdrant/fastembed)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-6B46C1)](https://github.com/BerriAI/litellm)
[![Tests](https://img.shields.io/badge/Tests-99%20passing-success?logo=pytest&logoColor=white)](#testing)
[![Status](https://img.shields.io/badge/Status-Phase%204%20Complete-success)](#roadmap)

**Repository:** [github.com/im-ad-45/DocuScale](https://github.com/im-ad-45/DocuScale)

---

## Overview

DocuScale is a modular retrieval-augmented generation service. **Retrieval runs entirely on the local machine** — ONNX embeddings, on-disk Qdrant, in-memory BM25, and a local cross-encoder, with no embedding API and no PyTorch/CUDA. **Generation is provider-agnostic**: any model supported by LiteLLM (Groq, OpenAI, local Ollama, ...) is selected through environment variables. The whole pipeline is exposed over an HTTP API with Server-Sent Events streaming and is covered by an offline test suite.

Capabilities added on top of the Phase 2 hybrid retriever:

- **HyDE query expansion** (optional): an LLM drafts a hypothetical answer passage to bridge vocabulary gaps on the dense path.
- **Grounded answer synthesis**: answers must cite `[doc_id#chunk_index]`; every citation is mechanically checked against the retrieved chunks, and out-of-domain questions are refused through a deterministic string contract.
- **Production API**: FastAPI service with document ingestion, blocking and streaming query endpoints, and a health check, all behind Pydantic v2 request/response contracts.
- **Offline test suite**: 99 pytest tests that need no network, API keys, or model downloads.

---

## Architecture

### Service layer

```text
                  Client (curl · httpx · fetch)
                                │  JSON over HTTP · SSE on /query/stream
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ FastAPI app · src/api/app.py                                   │
│ lifespan: build Services once, ensure collection, then serve   │
│                                                                │
│ routes.py   GET  /health · POST /ingest · POST /query          │
│             POST /query/stream  (SSE)                          │
│ schemas.py  Pydantic v2 request / response contracts           │
└───────────────────────────────┬────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ Services · src/api/services.py                                 │
│ VectorStore (locked) · HybridRetriever · Synthesizer           │
│ ingest lock: one writer at a time                              │
└───────────────────────────────┬────────────────────────────────┘
                                ├──▶ ingestion pipeline   (POST /ingest)
                                └──▶ query pipeline       (POST /query, POST /query/stream)
```

### Ingestion

```text
 Documents
     │
     ▼
 Token-aware chunker ── sliding window, configurable overlap, deterministic chunk IDs
     │
     ▼
 FastEmbed (BAAI/bge-small-en-v1.5, ONNX) ── dense vectors
     │
     ▼
 Local Qdrant (./qdrant_data) ── vectors + chunk payload, cosine distance
     │
     └──▶ BM25 index ── built in memory from the stored chunks (refreshed after every /ingest)
```

### Query pipeline

```text
                                 User Query
                                     │
                 ┌───────────────────┬───────────────────┐
                 ▼                                       ▼
 ┌───────────────────────────────┐                       │
 │ HyDE Expander (optional)      │                       │
 │ LLM drafts a hypothetical     │                       │ raw query, untouched
 │ answer passage. On failure,   │                       │
 │ falls back to the raw query   │                       │
 └───────────────────────────────┘                       │
                 │                                       │
                 ▼                                       ▼
 ┌───────────────────────────────┐       ┌───────────────────────────────┐
 │ Dense retrieval               │       │ Sparse retrieval              │
 │ embeds raw query + passage    │       │ BM25 (rank_bm25)              │
 │ FastEmbed → Qdrant (cosine)   │       │ searches the raw query only   │
 └───────────────────────────────┘       └───────────────────────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │ Reciprocal Rank Fusion        │
                     │ RRF, k = 60                   │
                     └───────────────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Cross-Encoder reranking       │
                     │ FastEmbed / ONNX, local CPU   │
                     │ scores (raw query, chunk)     │
                     └───────────────────────────────┘
                                     │  RetrievalResult(hits, hyde_passage)
                                     ▼
                     ┌───────────────────────────────┐
                     │ Synthesizer                   │
                     │ source-restricted prompt      │
                     │ ChatModel → LiteLLM           │
                     └───────────────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Citation audit                │
                     │ [doc#n] checked against the   │
                     │ retrieved chunk labels        │
                     │ exact-string refusal check    │
                     └───────────────────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Answer                        │
                     │ grounded, refused, citations  │
                     │ unknown_citations, sources    │
                     └───────────────────────────────┘
```

- **Dual path.** Dense and sparse retrieval run on different query strings when HyDE is on; with HyDE off both receive the raw query.
- **Single contract between stages.** `HybridRetriever.retrieve()` returns a `RetrievalResult` (`.hits`, `.hyde_passage`); the `Synthesizer` consumes `.hits`.
- **Two output modes, one pipeline.** `POST /query` calls `Synthesizer.answer()` and returns one audited JSON document. `POST /query/stream` calls `Synthesizer.stream()` for tokens, then `Synthesizer.audit()` on the joined text to produce the final `Answer`.

---

## Key Features

| Capability | Behavior |
|---|---|
| **Local embeddings** | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, CPU). No embedding API, no API keys. |
| **Local vector store** | `qdrant-client` in embedded on-disk mode (`./qdrant_data`); no Qdrant server. Cosine distance, persistent across runs. |
| **Hybrid retrieval** | Dense semantic search + BM25 lexical search, merged with Reciprocal Rank Fusion (`k = 60`). RRF operates on ranks, so the two score scales never need to be compared. |
| **Cross-encoder reranking** | Fused shortlist is rescored on (raw query, chunk) pairs with a local FastEmbed cross-encoder (`BAAI/bge-reranker-base` by default). |
| **HyDE (optional)** | Hypothetical answer passage embedded for the dense path only. Enabled with `DOCUSCALE_USE_HYDE=true`, `--hyde` on the CLI, or per request via `use_hyde`. |
| **Provider-agnostic LLM layer** | A `ChatModel` Protocol (`complete` + `stream`) implemented by a LiteLLM wrapper. Swap Groq / OpenAI / Ollama through config only. |
| **Citation audit** | `Synthesizer` extracts `[doc#n]` tags from the answer and verifies each against the retrieved chunk labels. |
| **Deterministic refusal** | The model must reply with an exact configured sentence when sources are insufficient; refusal is detected by string comparison, not LLM self-judgment. Empty retrieval skips the LLM call. |
| **HTTP API** | `/health`, `/ingest`, `/query`, `/query/stream`, with OpenAPI docs at `/docs`. Strict Pydantic v2 validation returns `422` on malformed input. |
| **SSE streaming** | Tokens stream in real time; the final `done` event carries the full citation audit. |
| **Idempotent ingestion** | Re-ingesting a `doc_id` replaces the document; stale chunks from a previous, longer version are deleted. |
| **Serialized store access** | Every call into the embedded Qdrant client runs under a re-entrant lock in `VectorStore`; ingestion requests are additionally serialized. The lock is never held during LLM calls. |
| **Offline tests** | Embedder, reranker, and LLM sit behind Protocols and are replaced with deterministic fakes in the suite. |
| **No orchestration framework** | No LangChain / LlamaIndex / CrewAI. Each stage is an explicit, individually testable module. |

### HyDE design decisions

| Decision | Rationale |
|---|---|
| Passage is embedded **only on the dense path** | A hypothetical answer is written in document vocabulary, which improves semantic recall. |
| BM25 receives the **untouched raw query** | LLM-generated text would inject unrequested terms into lexical matching (keyword drift). |
| Raw query is **prepended** to the passage before embedding | Anchors the vector to the user's intent if the LLM drifts off-topic. |
| Reranker scores against the **raw query** | Final relevance is judged against what the user actually asked. |
| LLM failure **falls back to standard retrieval** | HyDE errors (rate limits, timeouts) are logged at `WARNING` and never fail the request. `RetrievalResult.hyde_passage` is `None` in that case. |

### API design decisions

| Decision | Rationale |
|---|---|
| Services are built **once** in the app lifespan and injected (`create_app(services)`) | Models load at startup, not per request; tests inject fakes and skip model loading entirely. |
| `/query/stream` completes retrieval and **waits for the first token** before responding | Authentication and connectivity failures become a real HTTP `502` instead of a `200` stream containing an error. |
| The citation audit runs **after** streaming; the `done` event carries the verdict | The audit needs the complete text. Clients treat `grounded: false` as a signal to retract what was displayed. |
| `doc_id` is restricted to letters, digits, `_`, `.`, `-`, and spaces | `doc_id` is part of the citation label `doc_id#n`; characters such as `#`, `,`, `;`, `[` and `]` would corrupt citation parsing. |
| Provider failures return a generic `LLM generation failed` | Provider error text is logged server-side and never returned to clients. |

### Grounding contract

`Synthesizer.answer()` and the streaming `done` event both return an `Answer`:

| Field | Meaning |
|---|---|
| `answer` | The generated text. |
| `grounded` | Not refused, cites at least one real source, and cites no unknown source. |
| `refused` | The model returned the configured refusal sentence (or retrieval was empty). |
| `citations` | Cited labels that exist in the retrieved context. |
| `unknown_citations` | Cited labels that were **not** in the retrieved context (hallucinated). |
| `sources` | The chunks behind `citations`. |

The audit verifies that citations are **real**, not that a sentence is logically entailed by its cited chunk. See [Known Limitations](#known-limitations).

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Runtime |
| API | FastAPI + Uvicorn | HTTP service, OpenAPI docs, SSE streaming |
| Embeddings | FastEmbed, `BAAI/bge-small-en-v1.5` | Local ONNX dense embeddings |
| Vector store | `qdrant-client` (local mode) | Persistent vector storage and cosine search |
| Sparse search | `rank_bm25` | BM25 lexical matching |
| Rank fusion | Reciprocal Rank Fusion, `k = 60` | Merges dense and sparse rankings |
| Reranker | FastEmbed cross-encoder (`BAAI/bge-reranker-base`) | Local query–chunk relevance scoring |
| Query expansion | HyDE (`src/retrieval/hyde.py`) | LLM-drafted hypothetical passage for dense retrieval |
| LLM gateway | LiteLLM | Provider-neutral chat completions, blocking and streaming |
| Generation | `ChatModel` Protocol + `Synthesizer` | Source-restricted answers with audited citations |
| Configuration | Pydantic v2 + `python-dotenv` | Typed, immutable settings; `.env` loading |
| Chunking | Custom sliding window | Overlapping chunks with traceable metadata |
| Testing | pytest + httpx (`TestClient`) | Offline unit and integration suite |

---

## Project Layout

```text
DocuScale/
├── src/
│   ├── config.py                 # Pydantic v2 settings, prompt templates, env loading
│   ├── api/
│   │   ├── app.py                # app factory, lifespan, `app` for uvicorn
│   │   ├── routes.py             # /health, /ingest, /query, /query/stream (SSE)
│   │   ├── schemas.py            # Pydantic v2 request/response contracts
│   │   └── services.py           # Services container + production wiring
│   ├── ingestion/
│   │   └── chunker.py            # sliding-window chunker, deterministic chunk IDs
│   ├── storage/
│   │   └── vector_store.py       # locked local Qdrant manager: upsert, search, scroll, delete, count
│   ├── retrieval/
│   │   ├── bm25.py               # in-memory BM25 index
│   │   ├── fusion.py             # Reciprocal Rank Fusion
│   │   ├── reranker.py           # Reranker Protocol + FastEmbed cross-encoder
│   │   ├── hyde.py               # HyDE query expansion with failure fallback
│   │   └── hybrid.py             # HybridRetriever -> RetrievalResult
│   └── generation/
│       ├── llm.py                # ChatModel Protocol + LiteLLM client (complete / stream)
│       └── synthesizer.py        # grounded synthesis: answer(), stream(), audit()
├── tests/
│   ├── conftest.py               # fixtures: isolated settings, store, services, TestClient
│   ├── fakes.py                  # deterministic fake embedder, reranker, LLM
│   ├── corpus.py                 # shared test corpus
│   ├── test_config.py
│   ├── test_chunker.py
│   ├── test_fusion_bm25.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_api.py
├── run.py                        # CLI end-to-end verification against a live LLM
├── pytest.ini                    # pythonpath and test discovery
├── requirements.txt              # runtime dependencies
├── requirements-dev.txt          # runtime + pytest, httpx
├── .env.example
└── qdrant_data/                  # runtime-generated, git-ignored
```

---

## Requirements

- Python **3.10+** and `pip`
- Internet access on first run for dependencies and model downloads
- For generation, one of: a hosted-provider API key (e.g. Groq) **or** a running local [Ollama](https://ollama.com/) server

Retrieval itself needs no API keys. The reranker (`bge-reranker-base`) is the largest model download (about 1 GB); a lighter alternative can be set with `DOCUSCALE_RERANKER_MODEL`. The test suite needs none of this: it runs without model downloads, network access, or API keys.

---

## Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/im-ad-45/DocuScale.git
cd DocuScale
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

If PowerShell blocks script execution, allow it for the current session and activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
git clone https://github.com/im-ad-45/DocuScale.git
cd DocuScale
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env
```

`requirements-dev.txt` installs the runtime dependencies plus pytest and httpx. For a runtime-only install, use `requirements.txt`.

---

## Configuration

Edit `.env`. Every field of `Settings` can be overridden as `DOCUSCALE_<FIELD_NAME>`.

```env
# LLM: any LiteLLM model string (tested: groq/openai/gpt-oss-20b)
DOCUSCALE_LLM_MODEL=groq/openai/gpt-oss-20b
GROQ_API_KEY=your-groq-key

# HyDE query expansion (can also be set per request with "use_hyde", or per CLI run with --hyde)
DOCUSCALE_USE_HYDE=false
```

### LLM providers

| Provider | `DOCUSCALE_LLM_MODEL` | Credentials |
|---|---|---|
| Groq | `groq/<model>` (tested: `groq/openai/gpt-oss-20b`) | `GROQ_API_KEY` |
| Ollama (local) | `ollama_chat/<model>`, e.g. `ollama_chat/llama3.1` | none; set `DOCUSCALE_LLM_API_BASE=http://localhost:11434` |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |

`DOCUSCALE_LLM_API_KEY` is an optional provider-neutral key override. API keys are held as `SecretStr` and are not printed in settings output or returned by `/health`. Never commit `.env`.

### Retrieval and generation settings

| Variable | Default | Description |
|---|---|---|
| `DOCUSCALE_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed dense model |
| `DOCUSCALE_RERANKER_MODEL` | `BAAI/bge-reranker-base` | FastEmbed cross-encoder |
| `DOCUSCALE_CHUNK_SIZE` / `DOCUSCALE_CHUNK_OVERLAP` | `200` / `40` | Chunk window and overlap, in whitespace tokens (applies to `/ingest`) |
| `DOCUSCALE_CANDIDATE_K` | `20` | Candidates per retriever; also the reranker shortlist size |
| `DOCUSCALE_RRF_K` | `60` | RRF smoothing constant |
| `DOCUSCALE_FINAL_TOP_K` | `5` | Chunks passed to generation (overridable per request with `top_k`) |
| `DOCUSCALE_USE_HYDE` | `false` | Default for HyDE query expansion (overridable per request) |
| `DOCUSCALE_LLM_TEMPERATURE` | `0.0` | Deterministic generation by default |
| `DOCUSCALE_LLM_MAX_TOKENS` | `600` | Answer length cap |
| `DOCUSCALE_LLM_TIMEOUT` | `60` | Seconds per LLM call |

Prompt templates (`hyde_prompt`, `answer_prompt`) and the refusal sentence (`refusal_message`) are also configurable settings.

---

## Quickstart

With the virtual environment activated and `.env` configured.

### 1. Run the test suite

```bash
pytest
```

Runs all 99 tests offline in about 14 seconds. See [Testing](#testing).

### 2. Launch the API server

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete.` — the embedding and reranker models load (and download on first run) before the server accepts requests. Interactive OpenAPI docs are served at `http://127.0.0.1:8000/docs`.

Run **one worker and without `--reload`**: embedded Qdrant is single-process, and the BM25 index lives in each worker's memory.

### 3. Call the API

```bash
# index a document
curl -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{
  "documents": [{"doc_id": "fatf", "text": "FATF Recommendation 16, known as the travel rule, requires financial institutions to pass originator and beneficiary information along with wire transfers."}]
}'

# stream an answer (-N disables curl output buffering)
curl -N -X POST http://127.0.0.1:8000/query/stream -H "Content-Type: application/json" \
  -d '{"question": "What does FATF Recommendation 16 require?"}'
```

Full request and response contracts are in the [API Reference](#api-reference).

### 4. CLI end-to-end check (optional)

```bash
python run.py            # standard hybrid retrieval
python run.py --hyde     # HyDE-expanded dense retrieval
```

`run.py` indexes a small demo corpus, then runs four cases through retrieve → rerank → synthesize against the configured live LLM:

| Case | Pass condition |
|---|---|
| Three in-domain questions | Answer is `grounded` and cites the expected document |
| One out-of-domain question | Answer is `refused` |

Exit codes: `0` all checks passed, `1` one or more checks failed, `2` LLM authentication failed. Per-question output has this shape (values elided):

```text
Q: <question>
  HyDE passage: <only when HyDE is on and succeeded>
  Retrieved: <doc#n>, <doc#n>, <doc#n>
  A: <answer text with [doc#n] citations>
  cited=[...] unknown=[] grounded=True refused=False
  -> PASS
```

HyDE adds one LLM round trip per query; on rate-limited free tiers it doubles the call count.

### Programmatic use

```python
from src.config import load_settings
from src.generation.llm import LiteLLMClient
from src.generation.synthesizer import Synthesizer
from src.retrieval.hyde import HydeExpander
from src.retrieval.hybrid import HybridRetriever
from src.storage.vector_store import VectorStore

settings = load_settings()
llm = LiteLLMClient(settings)
store = VectorStore(settings)
store.create_collection()              # then upsert_documents(...) with chunked documents
retriever = HybridRetriever(store, settings, hyde=HydeExpander(llm, settings))
synthesizer = Synthesizer(llm, settings)

question = "What does FATF Recommendation 16 require?"
result = retriever.retrieve(question, use_hyde=True)    # RetrievalResult
answer = synthesizer.answer(question, result.hits)      # Answer
print(answer.answer, answer.grounded, answer.citations)
```

Call `retriever.refresh_index()` after writing documents directly to the store so the in-memory BM25 index stays in sync (the `/ingest` endpoint does this automatically).

---

## API Reference

Base URL: `http://127.0.0.1:8000`. All bodies are JSON. Schemas are defined in `src/api/schemas.py`, and the live OpenAPI spec is served at `/docs` and `/openapi.json`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Pipeline status, collection size, active model configuration |
| `POST /ingest` | Chunk, embed, and index documents; refresh the BM25 index |
| `POST /query` | One complete, citation-audited answer |
| `POST /query/stream` | The same answer streamed as Server-Sent Events |

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "collection": "docuscale",
  "collection_size": 12,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "reranker_model": "BAAI/bge-reranker-base",
  "llm_model": "groq/openai/gpt-oss-20b",
  "hyde_enabled": false
}
```

If the vector store is unreachable, the endpoint returns HTTP `503` with `"status": "degraded"` and `"collection_size": null`. Credentials are never included.

### `POST /ingest`

```bash
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "doc_id": "fatf",
        "text": "FATF Recommendation 16, known as the travel rule, requires financial institutions to pass originator and beneficiary information along with wire transfers.",
        "metadata": {"source": "demo"}
      },
      {
        "doc_id": "kyc",
        "text": "Know Your Customer rules require banks to verify the identity of clients before opening accounts."
      }
    ]
  }'
```

Example response:

```json
{ "documents": 2, "chunks_indexed": 2, "collection_size": 2 }
```

| Field | Rules |
|---|---|
| `documents` | 1–100 items; `doc_id` values must be unique within a request |
| `doc_id` | 1–200 characters from letters, digits, `_`, `.`, `-`, and space |
| `text` | 1–1,000,000 characters after trimming whitespace |
| `metadata` | Optional object stored with every chunk of the document |

Re-using an existing `doc_id` **replaces** that document: its previous chunks are deleted before the new ones are written. The BM25 index is refreshed before the response is returned.

### `POST /query`

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does FATF Recommendation 16 require?", "top_k": 3, "use_hyde": true}'
```

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | 1–2000 characters after trimming |
| `top_k` | integer | `DOCUSCALE_FINAL_TOP_K` | 1–50 chunks passed to generation |
| `use_hyde` | boolean | `DOCUSCALE_USE_HYDE` | Per-request HyDE override |
| `rerank` | boolean | `true` | Set `false` to skip the cross-encoder |

Example response (abridged):

```json
{
  "question": "What does FATF Recommendation 16 require?",
  "answer": {
    "answer": "It requires financial institutions to pass originator and beneficiary information along with wire transfers [fatf#0].",
    "refused": false,
    "grounded": true,
    "citations": ["fatf#0"],
    "unknown_citations": [],
    "sources": [
      { "chunk_id": "...", "doc_id": "fatf", "text": "...", "chunk_index": 0,
        "start_token": 0, "end_token": 24, "metadata": { "source": "demo" } }
    ]
  },
  "retrieval": {
    "hits": [
      { "chunk": { "...": "..." }, "rrf_score": 0.0328, "dense_rank": 1,
        "bm25_rank": 1, "rerank_score": 4.1 }
    ],
    "hyde_passage": "..."
  }
}
```

`retrieval.hyde_passage` is `null` when HyDE was off or its LLM call failed. `answer` follows the [grounding contract](#grounding-contract).

### `POST /query/stream`

Same request body as `/query`. The response is `text/event-stream`.

```bash
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does FATF Recommendation 16 require?"}'
```

| Event | Payload | When |
|---|---|---|
| `token` | `{"text": "<fragment>"}` | Repeated, as the LLM produces text |
| `done` | The full audited `Answer` (`answer`, `refused`, `grounded`, `citations`, `unknown_citations`, `sources`) | Once, after the last token |
| `error` | `{"detail": "LLM generation failed"}` | The LLM stream broke after it had started; no `done` follows |

Example wire format (values abridged):

```text
event: token
data: {"text": "It requires "}

event: token
data: {"text": "financial institutions to pass originator and beneficiary information [fatf#0]."}

event: done
data: {"answer":"It requires financial institutions to pass originator and beneficiary information [fatf#0].","refused":false,"grounded":true,"citations":["fatf#0"],"unknown_citations":[],"sources":[...]}
```

Streaming semantics:

- **Tokens are unaudited.** The citation audit needs the complete text, so the verdict arrives only in `done`. Treat `grounded: false` as a signal to retract what was displayed, or use `POST /query` when an answer must be verified before it is shown.
- **Failures before the first token are real HTTP errors.** Retrieval finishes and the first token is awaited before the response starts, so a bad API key or an unreachable provider returns `502`, not a `200` stream.
- **Empty retrieval** streams the configured refusal sentence as a single `token`, followed by a `done` with `refused: true`, without calling the LLM.

Python consumer:

```python
import json
import httpx

body = {"question": "What does FATF Recommendation 16 require?"}
with httpx.stream("POST", "http://127.0.0.1:8000/query/stream", json=body, timeout=None) as r:
    event = None
    for line in r.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:].strip())
            if event == "token":
                print(data["text"], end="", flush=True)
            elif event == "done":
                print("\n", {k: data[k] for k in ("grounded", "refused", "citations", "unknown_citations")})
            elif event == "error":
                print("\n[stream error]", data["detail"])
```

The browser `EventSource` API supports only `GET`, so browser clients should consume this `POST` endpoint with `fetch()` and a streamed response body.

### Status codes

| Code | When |
|---|---|
| `200` | Success |
| `422` | Request fails schema validation (empty `question`, `top_k` outside 1–50, invalid or duplicate `doc_id`, blank `text`, ...) |
| `502` | The LLM call failed: `/query`, or `/query/stream` before the first token. Body: `{"detail": "LLM generation failed"}` |
| `503` | `/health` only: the vector store is unreachable |

### Windows PowerShell

In PowerShell, `curl` is an alias for `Invoke-WebRequest`. Call `curl.exe` and pass the body from a file to avoid quoting problems:

```powershell
'{"question": "What does FATF Recommendation 16 require?"}' | Set-Content -Encoding ascii query.json
curl.exe -N -X POST http://127.0.0.1:8000/query/stream -H "Content-Type: application/json" --data-binary "@query.json"
```

---

## Testing

```bash
pytest                                 # full suite: 99 tests, ~14 s
pytest tests/test_api.py -k stream     # streaming contract only
pytest -x tests/test_retrieval.py      # stop at the first retrieval failure
```

`pytest.ini` sets `pythonpath = .` and `testpaths = tests`.

### Coverage

| File | Covers |
|---|---|
| `test_chunker.py` | Window arithmetic, overlap, tail handling, blank input, deterministic IDs, validation |
| `test_fusion_bm25.py` | RRF scoring and ordering, BM25 tokenization and ranking, empty index |
| `test_vector_store.py` | Collection lifecycle, idempotent upsert, dense-search round trip, per-document delete, scroll pagination, injectable embedder |
| `test_retrieval.py` | Hybrid ranking, per-stage evidence, reranker ordering, HyDE routing and fallback, BM25 refresh |
| `test_generation.py` | Citation parsing, grounded / hallucinated / uncited / refused answers, stream-vs-blocking audit parity, LiteLLM request plumbing |
| `test_api.py` | Every route: `422` validation, replace-on-reingest, query and refusal behavior, SSE event contract, `502` paths, `/health` degradation |
| `test_config.py` | Environment coercion, secret handling, validation, immutability |

### How the suite stays offline

The embedder, reranker, and LLM are injected through Protocols (`Embedder`, `Reranker`, `ChatModel`) and replaced with deterministic fakes: a hashed bag-of-words embedder, a word-overlap reranker, and a scriptable LLM that can fail before or during a stream. Everything else is real: embedded Qdrant in a temporary directory, BM25, RRF, FastAPI routing through `TestClient`, and the LiteLLM request path (exercised through LiteLLM's mock-response mode). Only model weights and the network are substituted, so the suite needs no downloads, API keys, or GPU.

---

## Known Limitations

- **Citation audit checks existence, not entailment.** A real label attached to an unsupported sentence passes the audit.
- **Streamed tokens are shown before they are audited.** The verdict arrives in the final `done` event.
- **Single process.** Embedded Qdrant is locked to one process and the BM25 index is per-process, so run one Uvicorn worker. Multi-worker deployments would need a Qdrant server and a shared lexical index.
- **BM25 is in-memory** and is rebuilt from the full collection on every `/ingest` request; suited to small and medium corpora.
- **No authentication, rate limiting, or CORS configuration.** Place the service behind a gateway before exposing it beyond localhost.
- **Whitespace tokens approximate model tokens** for chunk sizing.
- **Refusal on out-of-domain questions depends on the model following the prompt;** the retriever always returns top-k chunks. Smaller local models may refuse less reliably.
- **No retrieval benchmark yet.** Quality has been verified with sanity cases and tests, not measured metrics (see Roadmap).

---

## Roadmap

### Phase 1 — Ingestion, Embedding & Local Vector Indexing

- [x] Token-aware sliding-window chunking with configurable overlap
- [x] FastEmbed `BAAI/bge-small-en-v1.5` ONNX embeddings
- [x] Local Qdrant persistence and cosine search

### Phase 2 — Hybrid Retrieval & Cross-Encoder Reranking

- [x] BM25 lexical retrieval
- [x] Dense vector retrieval
- [x] Reciprocal Rank Fusion (RRF)
- [x] Cross-encoder reranking

### Phase 3 — Query Expansion & Grounded Generation

- [x] HyDE query expansion, toggled by `--hyde` / `DOCUSCALE_USE_HYDE`
- [x] Dense-only HyDE routing with raw-query prepend and graceful fallback
- [x] Provider-agnostic `ChatModel` Protocol with LiteLLM backend
- [x] Source-restricted synthesis with `[doc#n]` citations
- [x] Mechanical citation audit (`grounded` / `unknown_citations`)
- [x] Deterministic refusal contract
- [x] `RetrievalResult` retrieval output (`.hits`, `.hyde_passage`)

### Phase 4 — Production API & Streaming

- [x] FastAPI service layer (`src/api/`: app factory, routes, schemas, services)
- [x] Pydantic v2 contracts for all request and response payloads
- [x] Streaming Server-Sent Events (SSE) answer endpoint with end-of-stream citation audit
- [x] Blocking `/query` endpoint with the same audit
- [x] Document ingestion API with replace-on-reingest and automatic BM25 refresh
- [x] Health check endpoint
- [x] Serialized, thread-safe access to the embedded vector store
- [x] 99-test offline pytest suite

### Backlog — Evaluation

- [ ] Retrieval benchmark dataset
- [ ] Recall@K / Precision@K
- [ ] MRR / NDCG
- [ ] Latency and throughput profiling

### Backlog — Production hardening

- [ ] Authentication and rate limiting
- [ ] Multi-worker deployment (Qdrant server mode and a shared lexical index)

---

## Current Status

**Phase 4 is complete and verified.** All 99 tests pass, and live SSE streaming with the citation audit has been validated end to end.

DocuScale now serves the full local-retrieval, provider-agnostic-generation pipeline over HTTP:

```text
Client → FastAPI → [HyDE] → Dense ∥ BM25 → RRF (k=60) → Cross-Encoder → Synthesizer → Citation Audit → JSON | SSE
```

All four planned phases are done. Next focus: the evaluation backlog.

---

## Engineering Principles

- **Local-first.** Retrieval works without cloud infrastructure; the LLM provider is a configuration choice.
- **Modular.** Chunking, storage, retrieval, generation, API, and configuration are independently replaceable.
- **Lightweight.** No orchestration frameworks; dependencies are added only where a focused implementation is not enough.
- **Verifiable.** Citations are audited and refusals are deterministic, so behavior can be checked mechanically.
- **Testable.** Every external dependency sits behind a Protocol, so the whole system runs offline under test.
- **Incremental.** Each phase leaves a working system and a clean commit.

---

<p align="center"><strong>Engineered by: <a href="https://github.com/im-ad-45">Aditya Shukla</a></strong></p>
