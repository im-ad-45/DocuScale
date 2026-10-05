# DocuScale

> **A local-first hybrid RAG engine: BM25 + dense retrieval, cross-encoder reranking, optional HyDE query expansion, and citation-audited answer generation — built without orchestration frameworks.**

**Phase 3 Complete:** Ingestion → Chunking → Local Embedding → Qdrant → [HyDE] → BM25 ∥ Dense → RRF → Cross-Encoder Reranking → Grounded Synthesis with Citation Audit

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed%20%7C%20ONNX-0A7EA4)](https://github.com/qdrant/fastembed)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-FF4F64)](https://qdrant.tech/)
[![Pydantic](https://img.shields.io/badge/Config-Pydantic%20v2-E92063)](https://docs.pydantic.dev/)
[![Cross-Encoder](https://img.shields.io/badge/Reranker-Cross--Encoder%20%7C%20FastEmbed-2E8B57)](https://github.com/qdrant/fastembed)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-6B46C1)](https://github.com/BerriAI/litellm)
[![Status](https://img.shields.io/badge/Status-Phase%203%20Complete-success)](#roadmap)

**Repository:** [github.com/im-ad-45/DocuScale](https://github.com/im-ad-45/DocuScale)

---

## Overview

DocuScale is a modular retrieval-augmented generation pipeline. **Retrieval runs entirely on the local machine** — ONNX embeddings, on-disk Qdrant, in-memory BM25, and a local cross-encoder, with no embedding API and no PyTorch/CUDA. **Generation is provider-agnostic**: any model supported by LiteLLM (Groq, OpenAI, local Ollama, ...) is selected through environment variables.

Phase 3 builds on the Phase 2 hybrid retriever with:

- **HyDE query expansion** (optional): an LLM drafts a hypothetical answer passage to bridge vocabulary gaps on the dense path.
- **Grounded answer synthesis**: answers must cite `[doc_id#chunk_index]`; every citation is mechanically checked against the retrieved chunks, and out-of-domain questions are refused through a deterministic string contract.

---

## Architecture

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
     └──▶ BM25 index ── built in memory from the stored chunks (refresh_index() after ingesting)
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

---

## Key Features

| Capability | Behavior |
|---|---|
| **Local embeddings** | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, CPU). No embedding API, no API keys. |
| **Local vector store** | `qdrant-client` in embedded on-disk mode (`./qdrant_data`); no Qdrant server. Cosine distance, persistent across runs. |
| **Hybrid retrieval** | Dense semantic search + BM25 lexical search, merged with Reciprocal Rank Fusion (`k = 60`). RRF operates on ranks, so the two score scales never need to be compared. |
| **Cross-encoder reranking** | Fused shortlist is rescored on (raw query, chunk) pairs with a local FastEmbed cross-encoder (`BAAI/bge-reranker-base` by default). |
| **HyDE (optional)** | Hypothetical answer passage embedded for the dense path only. Enabled with `--hyde` or `DOCUSCALE_USE_HYDE=true`. |
| **Provider-agnostic LLM layer** | A single-method `ChatModel` Protocol, implemented by a LiteLLM wrapper. Swap Groq / OpenAI / Ollama through config only. |
| **Citation audit** | `Synthesizer` extracts `[doc#n]` tags from the answer and verifies each against the retrieved chunk labels. |
| **Deterministic refusal** | The model must reply with an exact configured sentence when sources are insufficient; refusal is detected by string comparison, not LLM self-judgment. Empty retrieval skips the LLM call. |
| **No orchestration framework** | No LangChain / LlamaIndex / CrewAI. Each stage is an explicit, individually testable module. |

### HyDE design decisions

| Decision | Rationale |
|---|---|
| Passage is embedded **only on the dense path** | A hypothetical answer is written in document vocabulary, which improves semantic recall. |
| BM25 receives the **untouched raw query** | LLM-generated text would inject unrequested terms into lexical matching (keyword drift). |
| Raw query is **prepended** to the passage before embedding | Anchors the vector to the user's intent if the LLM drifts off-topic. |
| Reranker scores against the **raw query** | Final relevance is judged against what the user actually asked. |
| LLM failure **falls back to standard retrieval** | HyDE errors (rate limits, timeouts) are logged at `WARNING` and never fail the request. `RetrievalResult.hyde_passage` is `None` in that case. |

### Grounding contract

`Synthesizer.answer()` returns an `Answer`:

| Field | Meaning |
|---|---|
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
| Embeddings | FastEmbed, `BAAI/bge-small-en-v1.5` | Local ONNX dense embeddings |
| Vector store | `qdrant-client` (local mode) | Persistent vector storage and cosine search |
| Sparse search | `rank_bm25` | BM25 lexical matching |
| Rank fusion | Reciprocal Rank Fusion, `k = 60` | Merges dense and sparse rankings |
| Reranker | FastEmbed cross-encoder (`BAAI/bge-reranker-base`) | Local query–chunk relevance scoring |
| Query expansion | HyDE (`src/retrieval/hyde.py`) | LLM-drafted hypothetical passage for dense retrieval |
| LLM gateway | LiteLLM | Provider-neutral chat completions |
| Generation | `ChatModel` Protocol + `Synthesizer` | Source-restricted answers with audited citations |
| Configuration | Pydantic v2 + `python-dotenv` | Typed, immutable settings; `.env` loading |
| Chunking | Custom sliding window | Overlapping chunks with traceable metadata |

---

## Project Layout

```text
DocuScale/
├── src/
│   ├── config.py                 # Pydantic v2 settings, prompt templates, env loading
│   ├── ingestion/
│   │   └── chunker.py            # sliding-window chunker, deterministic chunk IDs
│   ├── storage/
│   │   └── vector_store.py       # local Qdrant manager: upsert, dense search, full scroll
│   ├── retrieval/
│   │   ├── bm25.py               # in-memory BM25 index
│   │   ├── fusion.py             # Reciprocal Rank Fusion
│   │   ├── reranker.py           # FastEmbed cross-encoder reranker
│   │   ├── hyde.py               # HyDE query expansion with failure fallback
│   │   └── hybrid.py             # HybridRetriever -> RetrievalResult
│   └── generation/
│       ├── llm.py                # ChatModel Protocol + LiteLLM client
│       └── synthesizer.py        # grounded synthesis, citation audit, refusal handling
├── run.py                        # end-to-end verification entry point
├── .env.example
├── requirements.txt
└── qdrant_data/                  # runtime-generated, git-ignored
```

---

## Requirements

- Python **3.10+** and `pip`
- Internet access on first run for dependencies and model downloads
- For generation, one of: a hosted-provider API key (e.g. Groq) **or** a running local [Ollama](https://ollama.com/) server

Retrieval itself needs no API keys. The reranker (`bge-reranker-base`) is the largest model download (about 1 GB); a lighter alternative can be set with `DOCUSCALE_RERANKER_MODEL`.

---

## Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/im-ad-45/DocuScale.git
cd DocuScale
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
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
pip install -r requirements.txt
cp .env.example .env
```

---

## Configuration

Edit `.env`. Every field of `Settings` can be overridden as `DOCUSCALE_<FIELD_NAME>`.

```env
# LLM: any LiteLLM model string (tested: groq/openai/gpt-oss-20b)
DOCUSCALE_LLM_MODEL=groq/openai/gpt-oss-20b
GROQ_API_KEY=your-groq-key

# HyDE query expansion (can also be forced per run with --hyde)
DOCUSCALE_USE_HYDE=false
```

### LLM providers

| Provider | `DOCUSCALE_LLM_MODEL` | Credentials |
|---|---|---|
| Groq | `groq/<model>` (tested: `groq/openai/gpt-oss-20b`) | `GROQ_API_KEY` |
| Ollama (local) | `ollama_chat/<model>`, e.g. `ollama_chat/llama3.1` | none; set `DOCUSCALE_LLM_API_BASE=http://localhost:11434` |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |

`DOCUSCALE_LLM_API_KEY` is an optional provider-neutral key override. API keys are held as `SecretStr` and are not printed in settings output. Never commit `.env`.

### Retrieval and generation settings

| Variable | Default | Description |
|---|---|---|
| `DOCUSCALE_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed dense model |
| `DOCUSCALE_RERANKER_MODEL` | `BAAI/bge-reranker-base` | FastEmbed cross-encoder |
| `DOCUSCALE_CHUNK_SIZE` / `DOCUSCALE_CHUNK_OVERLAP` | `200` / `40` | Chunk window and overlap, in whitespace tokens |
| `DOCUSCALE_CANDIDATE_K` | `20` | Candidates per retriever; also the reranker shortlist size |
| `DOCUSCALE_RRF_K` | `60` | RRF smoothing constant |
| `DOCUSCALE_FINAL_TOP_K` | `5` | Chunks passed to generation |
| `DOCUSCALE_USE_HYDE` | `false` | Enable HyDE query expansion |
| `DOCUSCALE_LLM_TEMPERATURE` | `0.0` | Deterministic generation by default |
| `DOCUSCALE_LLM_MAX_TOKENS` | `600` | Answer length cap |
| `DOCUSCALE_LLM_TIMEOUT` | `60` | Seconds per LLM call |

Prompt templates (`hyde_prompt`, `answer_prompt`) and the refusal sentence (`refusal_message`) are also configurable settings.

---

## Usage

With the virtual environment activated:

```bash
python run.py            # standard hybrid retrieval
python run.py --hyde     # HyDE-expanded dense retrieval
```

`run.py` indexes a small demo corpus, then runs four end-to-end cases through retrieve → rerank → synthesize:

| Case | Pass condition |
|---|---|
| Three in-domain questions | Answer is `grounded` and cites the expected document |
| One out-of-domain question | Answer is `refused` |

Exit codes: `0` all checks passed, `1` one or more checks failed, `2` LLM authentication failed. Output per question has this shape (values elided):

```text
Q: <question>
  HyDE passage: <only when HyDE is on and succeeded>
  Retrieved: <doc#n>, <doc#n>, <doc#n>
  A: <answer text with [doc#n] citations>
  cited=[...] unknown=[] grounded=True refused=False
  -> PASS
```

HyDE adds one LLM round trip per query; on rate-limited free tiers, `--hyde` doubles the call count. The first run downloads the embedding and reranker models.

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
store = VectorStore(settings)          # after create_collection() / upsert_documents()
retriever = HybridRetriever(store, settings, hyde=HydeExpander(llm, settings))
synthesizer = Synthesizer(llm, settings)

question = "What does FATF Recommendation 16 require?"
result = retriever.retrieve(question, use_hyde=True)    # RetrievalResult
answer = synthesizer.answer(question, result.hits)      # Answer
print(answer.answer, answer.grounded, answer.citations)
```

Call `retriever.refresh_index()` after ingesting new documents so the in-memory BM25 index stays in sync with Qdrant.

---

## Known Limitations

- **Citation audit checks existence, not entailment.** A real label attached to an unsupported sentence passes the audit.
- **BM25 is in-memory.** It is rebuilt from Qdrant at startup and on `refresh_index()`; suited to small and medium corpora.
- **Whitespace tokens approximate model tokens** for chunk sizing.
- **Local Qdrant is single-process.** The on-disk store is locked to one process at a time.
- **Refusal on out-of-domain questions depends on the model following the prompt;** the retriever always returns top-k chunks. Smaller local models may refuse less reliably.
- **No retrieval benchmark yet.** Quality has been verified with sanity cases, not measured metrics (see Roadmap).

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

- [ ] FastAPI service layer
- [ ] Streaming Server-Sent Events (SSE) answer endpoint
- [ ] Document ingestion API
- [ ] Health check endpoint
- [ ] Pytest suite

### Backlog — Evaluation

- [ ] Retrieval benchmark dataset
- [ ] Recall@K / Precision@K
- [ ] MRR / NDCG
- [ ] Latency and throughput profiling

---

## Current Status

**Phase 3 is complete and verified.**

DocuScale now provides a full local-retrieval, provider-agnostic-generation pipeline:

```text
[HyDE] → Dense ∥ BM25 → RRF (k=60) → Cross-Encoder → Synthesizer → Citation Audit → Answer
```

Next: **Phase 4 — Production API & Streaming**.

---

## Engineering Principles

- **Local-first.** Retrieval works without cloud infrastructure; the LLM provider is a configuration choice.
- **Modular.** Chunking, storage, retrieval, generation, and configuration are independently replaceable.
- **Lightweight.** No orchestration frameworks; dependencies are added only where a focused implementation is not enough.
- **Verifiable.** Citations are audited and refusals are deterministic, so behavior can be checked mechanically.
- **Incremental.** Each phase leaves a working system and a clean commit.

---

<p align="center"><strong>Engineered by: <a href="https://github.com/im-ad-45">Aditya Shukla</a></strong></p>
