import asyncio
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.core.observability import traceable
from app.db.models.chunk import DocumentChunk
from app.db.models.document import Document
from app.documents.embedder import BgeM3Embedder
from app.prompts.sanitizer import sanitize_untrusted_text
from app.rag.reranker import BgeReranker
from app.rag.vector_store import RetrievedChunk


class HybridRetrievedChunk(RetrievedChunk):
    chunk_type: str = "text"
    extractor: str = "pymupdf"
    heading_path: Optional[str] = None
    citation_label: str = ""
    rrf_score: float = 0.0


def compute_sparse_dot_product(
    query_sparse: Dict[int, float],
    chunk_sparse: Dict[Any, float],
) -> float:
    """
    Compute exact lexical dot product between two sparse weight maps.
    Handles both integer and stringified JSONB keys.
    """
    if not query_sparse or not chunk_sparse:
        return 0.0
    total = 0.0
    for k, q_w in query_sparse.items():
        if k in chunk_sparse:
            total += float(q_w) * float(chunk_sparse[k])
        elif str(k) in chunk_sparse:
            total += float(q_w) * float(chunk_sparse[str(k)])
    return round(total, 6)


def compute_rrf_scores(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Standard Reciprocal Rank Fusion (RRF):
        score(d) = sum_{r in retrievers} 1 / (k + rank_r(d))
    where rank_r(d) is 1-indexed.
    Returns list of (chunk_id, rrf_score) sorted descending.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank_idx, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank_idx))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRAGRetriever:
    """
    Production 6-step Hybrid Retrieval Pipeline:
      Step 1: Query encoding via BAAI/bge-m3 (dense + sparse)
      Step 2: Parallel candidate retrieval (dense pgvector top-50 + sparse BM25 ts_rank_cd top-50),
              strictly tenant-isolated by user_id
      Step 3: Reciprocal Rank Fusion (k=60) -> top-20 candidates
      Step 4: Cross-encoder reranking via BAAI/bge-reranker-v2-m3
      Step 5: Truncate to top-5 with citation_label "[{doc_title}, p.{page_number}, {chunk_type}]"
      Step 6: Format sanitized context block with inline citation instructions
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Optional[BgeM3Embedder] = None,
        reranker: Optional[BgeReranker] = None,
    ):
        self.session = session
        self.embedder = embedder or BgeM3Embedder()
        self.reranker = reranker or BgeReranker()

    @traceable(name="step1_query_encoding", run_type="embedding")
    def _encode_query(self, query: str) -> Tuple[List[float], Dict[int, float]]:
        return self.embedder.encode_query(query)

    @traceable(name="step2_dense_retrieval", run_type="retriever")
    async def _dense_retrieve(
        self,
        query_dense: List[float],
        user_id: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = settings.RETRIEVAL_DENSE_TOPK,
    ) -> List[HybridRetrievedChunk]:
        if not user_id:
            return []

        cosine_dist = DocumentChunk.dense_embedding.cosine_distance(query_dense)
        stmt = (
            select(
                DocumentChunk,
                Document.filename,
                cosine_dist.label("distance"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                and_(
                    DocumentChunk.user_id == user_id,
                    Document.user_id == user_id,
                    DocumentChunk.dense_embedding.isnot(None),
                )
            )
        )
        if doc_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(doc_ids))

        stmt = stmt.order_by(cosine_dist.asc()).limit(top_k)

        try:
            res = await self.session.execute(stmt)
            rows = res.all()
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

        out: List[HybridRetrievedChunk] = []
        for chunk_obj, filename, dist in rows:
            sim = max(0.0, min(1.0, 1.0 - float(dist or 0.0)))
            out.append(self._to_hybrid_chunk(chunk_obj, filename, score=sim))
        return out

    @traceable(name="step2_sparse_retrieval", run_type="retriever")
    async def _sparse_retrieve(
        self,
        query: str,
        user_id: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = settings.RETRIEVAL_SPARSE_TOPK,
    ) -> List[HybridRetrievedChunk]:
        if not user_id or not query.strip():
            return []

        ts_query = func.plainto_tsquery("english", query)
        ts_rank = func.ts_rank_cd(DocumentChunk.content_tsv, ts_query)

        stmt = (
            select(
                DocumentChunk,
                Document.filename,
                ts_rank.label("bm25_score"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                and_(
                    DocumentChunk.user_id == user_id,
                    Document.user_id == user_id,
                    DocumentChunk.content_tsv.op("@@")(ts_query),
                )
            )
        )
        if doc_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(doc_ids))

        stmt = stmt.order_by(ts_rank.desc()).limit(top_k)

        try:
            res = await self.session.execute(stmt)
            rows = res.all()
        except Exception as e:
            logger.error(f"Sparse BM25 retrieval failed: {e}")
            return []

        out: List[HybridRetrievedChunk] = []
        for chunk_obj, filename, bm25_score in rows:
            out.append(self._to_hybrid_chunk(chunk_obj, filename, score=float(bm25_score or 0.0)))
        return out

    @staticmethod
    def _to_hybrid_chunk(
        chunk_obj: DocumentChunk,
        filename: str,
        score: float,
    ) -> HybridRetrievedChunk:
        ctype = getattr(chunk_obj, "chunk_type", None) or "text"
        ext = getattr(chunk_obj, "extractor", None) or "pymupdf"
        hpath = getattr(chunk_obj, "heading_path", None) or getattr(chunk_obj, "section_title", None)
        citation = f"[{filename}, p.{chunk_obj.page_number}, {ctype}]"
        return HybridRetrievedChunk(
            chunk_id=chunk_obj.id,
            document_id=chunk_obj.document_id,
            filename=filename,
            page_number=chunk_obj.page_number,
            content=sanitize_untrusted_text(chunk_obj.content),
            similarity_score=round(score, 4),
            metadata=chunk_obj.chunk_metadata or {},
            parent_chunk_id=chunk_obj.parent_chunk_id,
            section_title=chunk_obj.section_title,
            chunk_type=ctype,
            extractor=ext,
            heading_path=hpath,
            citation_label=citation,
        )

    @traceable(name="step3_rrf_fusion", run_type="chain")
    def _fuse_rrf(
        self,
        dense_candidates: List[HybridRetrievedChunk],
        sparse_candidates: List[HybridRetrievedChunk],
        k: int = settings.RRF_K,
        fused_top_k: int = settings.RRF_FUSED_TOPK,
    ) -> List[HybridRetrievedChunk]:
        by_id: Dict[str, HybridRetrievedChunk] = {}
        for c in dense_candidates + sparse_candidates:
            if c.chunk_id not in by_id:
                by_id[c.chunk_id] = c

        dense_ids = [c.chunk_id for c in dense_candidates]
        sparse_ids = [c.chunk_id for c in sparse_candidates]
        fused_ranks = compute_rrf_scores([dense_ids, sparse_ids], k=k)

        fused_chunks: List[HybridRetrievedChunk] = []
        for cid, rrf_sc in fused_ranks[:fused_top_k]:
            ch = by_id[cid]
            ch.rrf_score = round(rrf_sc, 6)
            fused_chunks.append(ch)
        return fused_chunks

    @traceable(name="step4_cross_encoder_rerank", run_type="chain")
    async def _rerank_candidates(
        self,
        query: str,
        candidates: List[HybridRetrievedChunk],
        top_k: int = settings.FINAL_TOPK,
    ) -> List[HybridRetrievedChunk]:
        reranked = await self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        return reranked[:top_k]  # type: ignore

    @traceable(name="hybrid_rag_retrieve", run_type="retriever")
    async def retrieve(
        self,
        query: str,
        user_id: str,
        document_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        top_k: int = settings.FINAL_TOPK,
    ) -> List[HybridRetrievedChunk]:
        """
        Execute the full 6-step hybrid retrieval pipeline for a tenant `user_id`.
        """
        if not user_id:
            raise ValueError("user_id is required for tenant-isolated retrieval.")

        scoped_doc_ids = list(doc_ids) if doc_ids else ([document_id] if document_id else None)

        # Step 1: Query encoding
        query_dense, _query_sparse = self._encode_query(query)

        # Step 2: Parallel dense + sparse candidate retrieval
        dense_candidates, sparse_candidates = await asyncio.gather(
            self._dense_retrieve(
                query_dense=query_dense,
                user_id=user_id,
                doc_ids=scoped_doc_ids,
                top_k=settings.RETRIEVAL_DENSE_TOPK,
            ),
            self._sparse_retrieve(
                query=query,
                user_id=user_id,
                doc_ids=scoped_doc_ids,
                top_k=settings.RETRIEVAL_SPARSE_TOPK,
            ),
        )

        if not dense_candidates and not sparse_candidates:
            return []

        # Step 3: RRF fusion (top-20)
        fused = self._fuse_rrf(
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            k=settings.RRF_K,
            fused_top_k=settings.RRF_FUSED_TOPK,
        )

        # Step 4 & 5: Cross-encoder rerank and truncate to top-5
        final_chunks = await self._rerank_candidates(
            query=query,
            candidates=fused,
            top_k=min(top_k, settings.FINAL_TOPK),
        )
        return final_chunks

    @staticmethod
    def format_context_block(chunks: List[HybridRetrievedChunk]) -> str:
        """
        Step 6 helper: builds sanitized context block with explicit citation tags.
        """
        if not chunks:
            return ""
        lines = []
        for ch in chunks:
            label = ch.citation_label or f"[{ch.filename}, p.{ch.page_number}, {ch.chunk_type}]"
            heading = f" | Heading: {ch.heading_path}" if ch.heading_path else ""
            lines.append(f"Source {label}{heading} (extractor={ch.extractor}):\n{sanitize_untrusted_text(ch.content)}")
        return "\n\n".join(lines)
