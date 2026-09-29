# DocuScale

> **A production-oriented local document retrieval foundation built for scalable semantic search — without cloud APIs, API keys, or heavyweight AI frameworks.**

**Phase 1 Complete:** Ingestion → Token-Aware Chunking → Local Embedding → Persistent Vector Indexing → Semantic Search

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed-ONNX)](https://github.com/qdrant/fastembed)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-FF4F64)](https://qdrant.tech/)
[![Pydantic](https://img.shields.io/badge/Config-Pydantic%20v2-E92063)](https://docs.pydantic.dev/)
[![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-success)](#roadmap)

**Repository:** [github.com/im-ad-45/DocuScale](https://github.com/im-ad-45/DocuScale)

---

## Overview

**DocuScale** is a modular document-ingestion and semantic-retrieval pipeline designed around a simple principle:

> Keep the retrieval foundation local, lightweight, deterministic, and independent of external AI APIs.

Phase 1 establishes the complete ingestion-to-vector-search path:

```text
                         DocuScale — Phase 1
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Documents                                                           │
│      │                                                               │
│      ▼                                                               │
│  ┌─────────────────────┐                                             │
│  │ Token-Aware Chunker │                                             │
│  │                     │                                             │
│  │ Sliding Window      │                                             │
│  │ + Configurable      │                                             │
│  │   Overlap           │                                             │
│  └──────────┬──────────┘                                             │
│             │                                                        │
│             ▼                                                        │
│  ┌─────────────────────┐                                             │
│  │ FastEmbed           │                                             │
│  │                     │                                             │
│  │ BAAI/bge-small-     │                                             │
│  │ en-v1.5             │                                             │
│  │ Quantized ONNX      │                                             │
│  └──────────┬──────────┘                                             │
│             │                                                        │
│             │ Embeddings                                             │
│             ▼                                                        │
│  ┌─────────────────────┐       Cosine Similarity                     │
│  │ Local Qdrant        │◄─────────────────────────────┐              │
│  │                     │                              │              │
│  │ ./qdrant_data       │                              │              │
│  │ Embedded / On-Disk  │                              │              │
│  └──────────┬──────────┘                              │              │
│             │                                         │              │
│             ▼                                         │              │
│      Semantic Search ─────────────────────────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Features

### Local-first embeddings with FastEmbed

DocuScale uses the quantized ONNX version of:

```text
BAAI/bge-small-en-v1.5
```

through **FastEmbed**.

This provides:

- Local embedding generation
- CPU-friendly ONNX inference
- No PyTorch dependency
- No CUDA requirement
- No external embedding API
- No API keys
- No per-token embedding charges
- Lower runtime overhead than a typical PyTorch-based embedding stack

The embedding model runs entirely on the machine executing DocuScale.

### Local Qdrant

Vector storage is provided by `qdrant-client` in local embedded/on-disk mode:

```text
./qdrant_data
```

This means Phase 1 does **not** require a separately deployed Qdrant server.

The vector index is persisted locally and can be reused between executions.

### No unnecessary orchestration framework

The Phase 1 implementation intentionally does not introduce LangChain, CrewAI, or similar orchestration layers.

The core pipeline is explicit:

```text
Chunk → Embed → Store → Search
```

This keeps the retrieval infrastructure:

- Lightweight
- Easy to inspect
- Easy to debug
- Easy to benchmark
- Easy to extend
- Free from unnecessary framework overhead

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Application/runtime |
| Embeddings | FastEmbed | Local embedding inference |
| Embedding Model | `BAAI/bge-small-en-v1.5` | Semantic vector generation |
| Inference | Quantized ONNX | Lightweight CPU inference |
| Vector Store | Qdrant Client | Local vector persistence/search |
| Similarity | Cosine | Semantic similarity metric |
| Configuration | Pydantic v2 | Typed configuration/data schemas |
| Environment Config | pydantic-settings | `.env` configuration |
| Chunking | Custom token-aware sliding window | Document segmentation |

---

## Directory Structure

```text
DocuScale/
│
├── src/
│   ├── config.py
│   │
│   ├── ingestion/
│   │   └── chunker.py
│   │
│   └── storage/
│       └── vector_store.py
│
├── run.py
├── .env.example
├── requirements.txt
│
└── qdrant_data/
    └── ...
```

### `src/config.py`

Central configuration and environment management using Pydantic v2 and `pydantic-settings`.

### `src/ingestion/chunker.py`

Contains the custom token-aware sliding-window text chunking implementation with configurable overlap.

### `src/storage/vector_store.py`

Handles local Qdrant vector storage and semantic similarity search.

### `run.py`

Standalone Phase 1 verification entry point.

It exercises the ingestion, embedding, persistence, and retrieval pipeline end-to-end.

### `.env.example`

Template for environment/configuration values used by the project.

### `requirements.txt`

Python dependencies required to install and run Phase 1.

### `qdrant_data/`

Local persistent Qdrant storage created by the application.

> `qdrant_data/` is runtime-generated storage and should not be treated as source code.

---

## Requirements

Before starting, ensure you have:

- Python **3.10 or newer**
- `pip`
- A terminal
- Internet access for the initial Python dependency/model download

After the embedding model is available locally, inference itself does not require an external embedding API.

---

## Installation

### Windows — PowerShell

#### 1. Clone the repository

```powershell
git clone https://github.com/im-ad-45/DocuScale.git
cd DocuScale
```

#### 2. Create a virtual environment

```powershell
python -m venv .venv
```

#### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Initialize environment configuration

```powershell
Copy-Item .env.example .env
```

---

### Linux / macOS

#### 1. Clone the repository

```bash
git clone https://github.com/im-ad-45/DocuScale.git
cd DocuScale
```

#### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

#### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

#### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Initialize environment configuration

```bash
cp .env.example .env
```

---

## Running Phase 1

With the virtual environment activated:

```bash
python run.py
```

The verification script performs the complete Phase 1 flow:

```text
Sample Documents
       │
       ▼
     Chunk
       │
       ▼
    Embed
       │
       ▼
 Persist to Qdrant
       │
       ▼
 Semantic Queries
       │
       ▼
Similarity Scores
```

On the first execution, FastEmbed may download the required embedding model before local inference begins.

---

## Data Flow

### 1. Ingestion

Raw document text enters the ingestion layer.

### 2. Chunking

The custom chunker divides the document into overlapping token-aware segments.

```text
Document
─────────────────────────────────────────────

Chunk A
[───────────────]

       Chunk B
       [───────────────]

              Chunk C
              [───────────────]
```

### 3. Embedding

Each chunk is transformed into a dense vector using:

```text
FastEmbed
    ↓
BAAI/bge-small-en-v1.5
    ↓
ONNX inference
```

### 4. Persistence

The resulting vectors and associated chunk data are stored in local Qdrant:

```text
./qdrant_data
```

### 5. Retrieval

A search query is embedded using the same embedding model.

Qdrant then compares the query vector against indexed vectors using cosine similarity.

```text
Query
  │
  ▼
Embedding
  │
  ▼
Query Vector
  │
  ▼
Cosine Similarity
  │
  ▼
Top Matching Chunks
```

---

## Roadmap

### Phase 1 — Ingestion, Embedding & Local Vector Indexing

- [x] Custom token-aware sliding-window chunking
- [x] Configurable chunk overlap
- [x] FastEmbed integration
- [x] `BAAI/bge-small-en-v1.5` ONNX embeddings
- [x] Local Qdrant persistence
- [x] Cosine similarity search
- [x] End-to-end verification script

### Phase 2 — Hybrid Retrieval

- [ ] BM25 lexical retrieval
- [ ] Dense vector retrieval
- [ ] Reciprocal Rank Fusion (RRF)
- [ ] Combined hybrid retrieval pipeline
- [ ] Retrieval evaluation and comparison

### Phase 3 — Retrieval Quality & Query Expansion

- [ ] Cross-Encoder reranking
- [ ] Top-K candidate reranking
- [ ] HyDE query expansion
- [ ] Improved retrieval relevance evaluation
- [ ] Retrieval pipeline optimization

### Phase 4 — Production API & Streaming

- [ ] FastAPI service layer
- [ ] Retrieval API endpoints
- [ ] Server-Sent Events (SSE) streaming
- [ ] Streaming retrieval/response pipeline
- [ ] Production-oriented API configuration

---

## Engineering Principles

DocuScale is being developed around several engineering principles:

### Local-first

Core retrieval functionality should work without mandatory cloud infrastructure.

### Modular

Chunking, storage, configuration, retrieval, and serving should remain independently replaceable components.

### Lightweight

Avoid introducing large dependencies where a focused implementation is sufficient.

### Observable

The pipeline should be easy to inspect from ingestion through retrieval.

### Incremental

Each phase establishes a working foundation for the next instead of introducing the entire RAG stack at once.

---

<p align="center">
  <strong>Engineered by: <a href="https://github.com/im-ad-45">Aditya Shukla</a></strong>
</p>
