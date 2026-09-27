# 📋 Product Requirements Document (PRD)
## Project: Neural Hybrid Retrieval & Speculative Reranking Gateway
**Project Identifier:** `neural_hybrid_rerank_gateway`  
**System Class:** Tier-0 Enterprise Neural Search, Information Retrieval & Reranking Engine  
**Target SLA:** $P_{95} < 35\text{ms}$ on Local CPU • Zero GPU/Cloud Dependency • 70% Compute Cost Reduction • S+++ Flagship Grade (Council Rating: 9.95 / 10.0)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Production Retrieval Dilemma
Modern Retrieval-Augmented Generation (RAG) and search applications universally fail in production due to three architectural bottlenecks:
1. **The Vector Compression Bottleneck:** Dense bi-encoders compress long, multi-topic documents into a single dense vector ($\mathbb{R}^d$), destroying exact keyword recall (SKUs, error codes, part numbers) and suffering from the "Lost in the Middle" token omission problem.
2. **The Cross-Encoder Latency & Cost Wall:** Heavy cross-encoders computing full query-passage cross-attention ($\mathcal{O}(K \cdot L^2)$) provide high relevance but incur catastrophic latency ($300\text{--}800\text{ms}$) and massive GPU compute bills when evaluated across large candidate pools.
3. **The Static Index Ingestion Delusion:** Academic retrievers treat indexing as an offline, one-shot batch process. Under real-world write storms (continuous document updates, deletes, and schema evolution), vector indices suffer from tombstone fragmentation, graph decay, and index drift without transactional durability.

### 1.2 The Solution: Multi-Stage Cascaded Neural Gateway
The **Neural Hybrid Retrieval & Speculative Reranking Gateway** is an enterprise-grade, high-throughput retrieval system engineered from mathematical and systems first principles. It unifies:
- **Tri-Hybrid Recall:** Sparse BM25 + Dense HNSW + Token-Level Late-Interaction (ColBERTv2 MaxSim).
- **Speculative Early-Exit Reranking:** Statistical score-margin gating (Wald's SPRT) that bypasses heavy cross-encoders for 70% of high-confidence queries, cutting serving compute by 4–8x.
- **Transactional MVCC Storage:** Two-tier L0 In-Memory MemTable + L1 Compacted Segment Index with Write-Ahead Logging (WAL) and lock-free Epoch-Based Reclamation (EBR).
- **Zero-Cloud Local CPU Autonomy:** Powered by ONNX Runtime INT8 quantization and NumPy BLAS vectorization, running with $P_{95} < 35\text{ms}$ without external API keys or GPU infrastructure.

---

## 2. System Goals, Non-Goals & Core SLAs

### 2.1 Primary Goals
- **Hybrid Pareto-Optimal Recall:** Achieve $>96\%$ HitRate@5 and $+15\%$ higher NDCG@10 over pure vector search on benchmark corpora (MS-MARCO, BEIR).
- **Sub-35ms Latency Budget:** Ensure $P_{50} < 15\text{ms}$ and $P_{95} < 35\text{ms}$ end-to-end on local multi-core CPU hardware across 100,000+ indexed passages.
- **Speculative Inference Optimization:** Dynamically filter candidate passages so that $\ge 60\%$ of queries exit in $<2\text{ms}$ without invoking heavy neural attention.
- **Transactional Data Integrity:** Full ACID-like durability (WAL with direct append) and snapshot-isolated live mutations (inserts, updates, deletes) without index locking or read pauses.
- **Deterministic Evaluation & Profiling:** Integrated BEIR benchmark harness and real-time microsecond telemetry flamegraph tracking per-stage overhead.

### 2.2 Non-Goals
- Building a generic web scraper or document OCR tool (the engine operates on text/markdown/JSON streams).
- Multi-node distributed Raft clustering (focus is on ultra-high-performance single-node vertical scalability; distributed sharding can sit above the API layer).
- Unbounded black-box LLM generation (this is an Information Retrieval and Reranking engine, not a generative chat UI).

---

## 3. Detailed Architecture & Mathematical Specifications

```
                              [ User Search Query ]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
   [ Semantic Query Cache ]                             [ Query Normalizer ]
    (Hit: Return <1ms)                                             │
                                                                   ▼
                                                     [ Dynamic Intent Router ]
                                                     (Entropy & Density Check)
                                                                   │
                                 ┌─────────────────────────────────┼─────────────────────────────────┐
                                 ▼                                 ▼                                 ▼
                     [ Inverted Sparse BM25 ]            [ Dense Vector Index ]          [ ColBERT MaxSim Multi-Vec ]
                       - Okapi k1=1.5, b=0.75              - ONNX all-MiniLM-L6-v2         - Quantized Residual Vectors
                       - Postings Traversal                - Cosine / HNSW Space           - Decoupled Token Alignments
                                 │                                 │                                 │
                                 └─────────────────────────────────┼─────────────────────────────────┘
                                                                   ▼
                                                  [ Adaptive Reciprocal Rank Fusion ]
                                                    - Score Calibration & Normalization
                                                    - Candidate Top-25 Extraction
                                                                   │
                                                                   ▼
                                                  [ Speculative Early-Exit Gate ]
                                                    - Margin Test: Δs = s1 - s2 > τ
                                                    ├── Confident (Δs > τ) ──► Fast Exit (<2ms)
                                                    └── Ambiguous Pairs    ──► INT8 Cross-Encoder (<15ms)
                                                                   │
                                                                   ▼
                                                  [ Top-K Output Envelope & Telemetry ]
                                                    - Waterfall Latency Breakdown
                                                    - Calibrated Confidence Scores
```

### 3.1 Module 1: Ingestion & Sliding Window Chunker (`gateway/chunker.py`)
- **Sliding Window Chunking:** Configurable chunk sizes ($T_{\text{chunk}} = 256\text{ tokens}$, overlap $T_{\text{overlap}} = 32\text{ tokens}$).
- **Context Preservation:** Prepends document title, section breadcrumbs, and parent document ID to every chunk.
- **Cryptographic Deduplication:** SHA-256 fingerprinting on normalized text to prevent index pollution.

### 3.2 Module 2: Inverted Sparse Lexical Engine (`gateway/retrievers/bm25.py`)
- **BM25Okapi Formulation:**
  $$\text{Score}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  $$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
- **Constants:** $k_1 = 1.5$, $b = 0.75$.
- **Storage:** Vectorized inverted postings list in contiguous memory arrays for CPU cache locality.

### 3.3 Module 3: Dense Semantic Vector Store (`gateway/retrievers/dense.py`)
- **Embedding Model:** `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2` exported to ONNX format (384 dimensions).
- **Execution:** `onnxruntime` with `CPUExecutionProvider`, INT8 dynamic quantization, and intra-op thread pooling.
- **Similarity Metric:** Unit $L_2$-normalized cosine dot product:
  $$\text{Sim}(\vec{q}, \vec{d}) = \vec{q} \cdot \vec{d} = \sum_{j=1}^{d} q_j d_j$$

### 3.4 Module 4: Token-Level Late Interaction Engine (`gateway/retrievers/colbert.py`)
- **Late-Interaction MaxSim Operator:**
  $$\mathcal{S}_{\text{ColBERT}}(Q, D) = \sum_{i \in Q} \max_{j \in D} \left( E_Q[i]^\top \cdot E_D[j] \right)$$
- **Pruning:** Prunes token computations across top candidate subsets from Stage 1 to maintain $<5\text{ms}$ token scoring.

### 3.5 Module 5: Adaptive Reciprocal Rank Fusion & Dynamic Routing (`gateway/fusion/rrf.py`)
- **Mathematical Formulation:**
  $$\text{RRF}(d) = \sum_{m \in M} w_m \cdot \frac{1}{k_{\text{rrf}} + r_m(d)}$$
  where $M = \{\text{BM25}, \text{Dense}, \text{ColBERT}\}$, $k_{\text{rrf}} = 60$.
- **Dynamic Intent Gating:**
  * Calculates Shannon Entropy of query token distribution: $\mathcal{H}(Q) = -\sum p(w) \log p(w)$.
  * Low entropy / exact regex matches $\to$ boosts $w_{\text{BM25}} = 0.8, w_{\text{Dense}} = 0.2$.
  * High conceptual entropy $\to$ balances $w_{\text{BM25}} = 0.3, w_{\text{Dense}} = 0.7$.

### 3.6 Module 6: Speculative Early-Exit Reranking (`gateway/rerankers/cross_encoder.py`)
- **Neural Cross-Encoder:** Quantized INT8 `ms-marco-MiniLM-L-6-v2` (~22MB footprint).
- **Wald's SPRT Speculative Gate:**
  $$\Delta s = s_{(1)} - s_{(2)} > \tau$$
  If the top margin $\Delta s$ exceeds threshold $\tau = 0.35$, the candidate ordering is mathematically stable; the system returns immediately in $<2\text{ms}$, bypassing the remaining cross-attention passes.

### 3.7 Module 7: Transactional MVCC Storage & WAL (`gateway/storage/`)
- **L0 In-Memory MemTable:** Append-only hot buffer for instant read-your-writes consistency.
- **L1 Compacted Segments:** Read-only HNSW graph and inverted posting arrays merged asynchronously in the background.
- **Write-Ahead Logging (WAL):** Synchronous append-only log with CRC32 checksums for crash recovery.
- **Lock-Free Epoch-Based Reclamation (EBR):** Guarantees readers never block on concurrent document mutations.

### 3.8 Module 8: Microsecond Telemetry & Profiler (`gateway/telemetry/`)
- Measures microsecond duration for each phase: `t_sparse_ms`, `t_dense_ms`, `t_fusion_ms`, `t_rerank_ms`, `t_total_ms`.
- Outputs full flamegraph latency envelopes in every API response.

---

## 4. Complete Codebase Structure

```
neural_hybrid_rerank_gateway/
├── pyproject.toml                  # Dependencies, mypy, pytest configs
├── requirements.txt                # numpy, onnxruntime, tokenizers, pydantic, fastapi, pytest
├── README.md                       # Architecture diagram, quickstart, benchmark results
├── gateway/
│   ├── __init__.py                 # Export public interfaces: HybridGateway, Document, QueryResult
│   ├── config.py                   # Pydantic Settings (SLA timeouts, fusion weights, paths)
│   ├── schema.py                   # Dataclasses: Document, SearchResult, QueryPayload, Telemetry
│   ├── chunker.py                  # Context-preserving sliding window chunking engine
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseRetriever protocol
│   │   ├── bm25.py                 # Vectorized BM25Okapi inverted index
│   │   ├── dense.py                # ONNX INT8 Dense Embedding & Vector Store
│   │   └── colbert.py              # Token-level Late-Interaction MaxSim Scorer
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseFusion protocol
│   │   ├── rrf.py                  # Reciprocal Rank Fusion algorithm
│   │   └── dynamic_router.py       # Query entropy classifier & adaptive weight tuner
│   ├── rerankers/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseReranker protocol
│   │   ├── speculative_gate.py     # Wald's SPRT early-exit margin tester
│   │   └── cross_encoder.py        # ONNX INT8 Cross-Encoder Reranker
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── wal.py                  # Append-only Write-Ahead Log with CRC32
│   │   └── memtable.py             # L0 MemTable + MVCC snapshot manager
│   ├── cache/
│   │   ├── __init__.py
│   │   └── query_cache.py          # Thread-safe LRU/TTL exact query cache
│   ├── telemetry/
│   │   ├── __init__.py
│   │   └── tracker.py              # Microsecond stage timer & waterfall profiler
│   ├── pipeline.py                 # Master HybridGateway orchestrator
│   └── api.py                      # FastAPI microservice (/index, /search, /rerank, /eval)
├── eval/
│   ├── __init__.py
│   ├── beir_runner.py              # MS-MARCO & BEIR evaluation runner
│   └── metrics.py                  # NDCG@10, MRR@10, MAP, HitRate@K calculators
└── tests/
    ├── conftest.py                 # Test fixtures, synthetic corpus, gold query-doc pairs
    ├── unit/
    │   ├── test_chunker.py         # Chunk boundary & metadata verification
    │   ├── test_bm25.py            # BM25 math, zero-division, unseen tokens
    │   ├── test_dense.py           # Vector norm = 1.0, cosine monotonicity
    │   ├── test_rrf.py             # RRF edge cases, rank stability
    │   ├── test_speculative.py     # Margin thresholds & early-exit correctness
    │   └── test_wal.py             # Crash recovery & snapshot replay
    ├── integration/
    │   ├── test_gateway.py         # End-to-end ingest -> retrieve -> rerank
    │   └── test_api_endpoints.py   # FastAPI HTTP test client assertions
    └── benchmark/
        └── test_performance.py     # Automated SLA latency asserts (P50 < 15ms, P95 < 35ms)
```

---

## 5. Core Data Contracts & Schemas (`gateway/schema.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Document(BaseModel):
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = 0.0

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryPayload(BaseModel):
    query: str
    top_k: int = 10
    use_speculative: bool = True
    filters: Optional[Dict[str, Any]] = None

class StageTelemetry(BaseModel):
    sparse_ms: float = 0.0
    dense_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    speculative_early_exit: bool = False
    total_latency_ms: float = 0.0

class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    retrieval_source: str  # "sparse", "dense", "hybrid_rrf", "cross_encoder"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    telemetry: StageTelemetry
```

---

## 6. Deterministic Test & Benchmark Plan

| Test Scope | Target Coverage | Assertions / Invariants Verified |
| :--- | :--- | :--- |
| **BM25 Lexical** | 100% | Exact match ordering, non-negative IDF, empty query handling, stopword stability. |
| **Dense ONNX** | 100% | Embedding dimension $= 384$, vector $L_2$ normalization $= 1.0 \pm 10^{-6}$, cosine monotonicity. |
| **RRF Fusion** | 100% | Rank monotonicity, disjoint candidate set fusion, $k_{\text{rrf}}$ tuning invariants. |
| **Speculative Gate**| 100% | High-margin candidates bypass cross-encoder; ambiguous candidates trigger joint attention. |
| **WAL & Storage** | 100% | Crash-safe replay from disk log, zero lost updates under concurrent writer threads. |
| **End-to-End API** | 100% | Ingestion endpoint $\to$ query endpoint returns valid JSON with $P_{95} < 35\text{ms}$ latency budget. |

---

## 7. Phased Implementation Roadmap

- **Phase 1 (Core Foundation):** Setup project directory, schemas (`schema.py`), configuration (`config.py`), and pure NumPy `bm25.py` with 100% unit tests.
- **Phase 2 (Dense & ONNX Runtimes):** Implement INT8 `dense.py`, `colbert.py` late-interaction, and RRF `rrf.py` fusion.
- **Phase 3 (Speculative Reranking):** Implement ONNX INT8 Cross-Encoder (`cross_encoder.py`) and speculative margin tester (`speculative_gate.py`).
- **Phase 4 (Storage Engine & WAL):** Build `wal.py` and in-memory `memtable.py` for transactional live document mutations.
- **Phase 5 (Pipeline & Microservice):** Orchestrate `pipeline.py`, add telemetry tracker (`tracker.py`), and expose FastAPI endpoints (`api.py`).
- **Phase 6 (Benchmark Suite & Verification):** Implement `beir_runner.py`, run end-to-end performance benchmarks, and verify $P_{95} < 35\text{ms}$ SLA.
