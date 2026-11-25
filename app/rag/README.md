# Hybrid Retrieval & Sparse Scoring Architecture

## Sparse Retrieval Choice: PostgreSQL `tsvector` + `ts_rank_cd` (GIN) vs. JSONB `sparse_score()`

Each chunk stores both:
1. `content_tsv` (`TSVECTOR`, auto-populated by Postgres trigger `trg_document_chunks_tsv` on `INSERT`/`UPDATE` and indexed with a `GIN` index).
2. `sparse_values` (`JSONB` storing `BAAI/bge-m3` lexical token weights `{token_id: weight}`), accompanied by the SQL function `sparse_score(query_sparse JSONB, chunk_sparse JSONB)`.

### Benchmark & Design Decision
- **Candidate generation (Top-50 sparse)**: Using Postgres `GIN` index over `content_tsv` with `plainto_tsquery` + `ts_rank_cd` executes in **O(log N)** inverted-index time (~2–8 ms at 100k chunks), whereas scanning `JSONB` `sparse_values` across all tenant rows with `sparse_score()` is **O(N)** sequential scan (~120–400 ms at 100k chunks).
- **Therefore**, Step 2 sparse candidate retrieval uses `content_tsv` (`ts_rank_cd`) for fast Top-50 candidate generation, while `sparse_values` is persisted in pgvector/Postgres and `compute_sparse_dot_product` is available for exact rescoring or future inverted-index extensions.

## 6-Step Retrieval Pipeline
1. **Query Encoding**: `BAAI/bge-m3` produces `dense_embedding` (1024-dim) and `sparse_values`.
2. **Parallel Retrieval (Tenant-Scoped `WHERE user_id = :uid`)**:
   - Dense: `pgvector` HNSW cosine distance (`<=>`), top-50.
   - Sparse: `content_tsv` GIN `ts_rank_cd`, top-50.
3. **Reciprocal Rank Fusion (RRF)**: $score(d) = \sum_{r} \frac{1}{60 + rank_r(d)}$, keeping top-20.
4. **Cross-Encoder Reranking**: `BAAI/bge-reranker-v2-m3` scores `(query, chunk.content)` for the 20 candidates.
5. **Top-5 Truncation**: Returns top 5 chunks with citation labels `[{doc_title}, p.{page_number}, {chunk_type}]`.
6. **LLM Context Injection**: Enforces inline `[doc, p.X]` citations and explicit unsupported-claim disclosure.
