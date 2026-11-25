# Student Assistant — Multi-Extractor Hybrid RAG Architecture

## (a) Adaptive Page-Aware Ingestion Pipeline

```
PDF Upload (POST /api/v1/documents/upload → 202 Accepted, status="queued")
    │
    ▼  (Celery + Redis async queue, SHA-256 + user_id dedup)
┌──────────────────────────────────────────────────────────┐
│ 1. Page-Level Triage (PyMuPDF single pass)               │
│    - Compute doc-wide mean_words_per_page                │
│    - Measure word_count, table_count, image_coverage_pct │
└──────┬───────────────┬──────────────────┬────────────────┘
       │               │                  │
       ▼               ▼                  ▼
  [pymupdf]        [docling]          [gemini]         [mixed]
  Text blocks     Table-heavy        Scanned /         Split text,
  (reading order) (Markdown;         Diagrams @        tables & figures
                   Marker/Unstr.      200 DPI +        in reading order
                   fallback)          JSON schema
       │               │                  │                │
       └───────────────┴──────────────────┴────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│ 2. Adaptive Chunker                                      │
│    - Text: 512 tokens, 64 overlap, cap 1024, sentence-safe│
│    - Table: 1 table = 1 chunk (row-split w/ header repeat)│
│    - Figure: intact Gemini description + caption         │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│ 3. Bi-Encoder Embedding (BAAI/bge-m3)                    │
│    - dense_embedding: vector(1024)                       │
│    - sparse_values: {token_id: weight}                   │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│ 4. PostgreSQL + pgvector Storage (status="ready")        │
│    - HNSW index on dense_embedding (m=16, ef_constr=200) │
│    - GIN index on content_tsv (auto-populated trigger)   │
│    - B-tree on (user_id, document_id, page_number)       │
└──────────────────────────────────────────────────────────┘
```

## (b) 6-Step Hybrid Retrieval Pipeline

```
User Query
    │
    ▼
[1. BAAI/bge-m3 Query Encoder] ──► dense_vec (1024) + sparse_weights
    │
    ├───────────────────────────────┐
    ▼ (WHERE user_id = :uid)        ▼ (WHERE user_id = :uid)
[2a. Dense pgvector Top-50]     [2b. Sparse BM25 Top-50]
(cosine distance <=>)           (content_tsv ts_rank_cd)
    │                               │
    └───────────────┬───────────────┘
                    ▼
[3. Reciprocal Rank Fusion (RRF, k=60)] ──► Top-20 Fused Candidates
                    │
                    ▼
[4. Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)]
                    │
                    ▼
[5. Top-5 Truncation] ──► [{doc_title}, p.{page_number}, {chunk_type}]
                    │
                    ▼
[6. Tutor / Aggregator LLM] (Inline [doc, p.X] citations + LangSmith trace)
```
