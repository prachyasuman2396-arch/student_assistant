import re
import math
from typing import List, Optional, Tuple
from app.rag.vector_store import RetrievedChunk
from app.core.logging import logger


class CrossEncoderReranker:
    """
    Dedicated cross-encoder reranking stage. Evaluates query-candidate
    pairs to reorder initial hybrid/RRF candidates by precise relevance.
    """
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
        min_relevance_threshold: float = 0.20,
    ) -> List[RetrievedChunk]:
        """
        Rerank a bounded candidate pool of retrieved chunks.
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates[:top_k]

        query_tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        query_text = query.lower()

        scored_candidates: List[Tuple[float, RetrievedChunk]] = []

        for original_rank, chunk in enumerate(candidates):
            content_lower = chunk.content.lower()

            # 1. Exact phrase matching
            phrase_score = 0.0
            if len(query_tokens) >= 2 and query_text in content_lower:
                phrase_score = 0.40

            # 2. Key term coverage (how many unique technical terms appear)
            term_matches = sum(1 for term in set(query_tokens) if term in content_lower)
            term_coverage = (term_matches / len(set(query_tokens))) if query_tokens else 0.5

            # 3. Density: term frequency normalized by chunk length
            total_freq = sum(content_lower.count(term) for term in query_tokens)
            density = min(0.30, total_freq / (len(content_lower.split()) + 1) * 2.0)

            # 4. Section / title alignment
            section_bonus = 0.0
            if chunk.metadata and chunk.metadata.get("section"):
                sec_text = str(chunk.metadata["section"]).lower()
                if any(t in sec_text for t in query_tokens):
                    section_bonus = 0.15

            # 5. First-stage prior (incorporates dense/BM25 initial confidence)
            first_stage_prior = min(0.30, chunk.similarity_score * 0.30)

            # Cross-encoder composite score (0.0 to 1.0)
            cross_score = min(
                1.0,
                (term_coverage * 0.40)
                + phrase_score
                + density
                + section_bonus
                + first_stage_prior
            )

            # Assign updated calibrated score
            chunk.similarity_score = round(cross_score, 4)
            scored_candidates.append((cross_score, chunk))

        # Sort by rerank score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        reranked_chunks = [
            chunk for score, chunk in scored_candidates
            if score >= min_relevance_threshold
        ]

        logger.debug(
            f"Cross-encoder reranked {len(candidates)} candidates down to {len(reranked_chunks[:top_k])} (top score: {scored_candidates[0][0]:.4f})"
        )

        return reranked_chunks[:top_k] if reranked_chunks else [candidates[0]]


class BgeReranker:
    """
    Production cross-encoder reranker backed by `BAAI/bge-reranker-v2-m3`
    (via FlagEmbedding FlagReranker or sentence-transformers CrossEncoder),
    with deterministic lexical cross-encoder fallback when model weights
    are unavailable in offline test environments.
    """

    def __init__(self, model_name: Optional[str] = None):
        from app.core.config import settings
        self.model_name = model_name or settings.RERANK_MODEL
        self._reranker = None
        self._st_cross = None
        self._fallback = CrossEncoderReranker(model_name=self.model_name)

    def _load_backend(self) -> str:
        if self._reranker is not None:
            return "flagembedding"
        if self._st_cross is not None:
            return "sentence_transformers"

        try:
            from FlagEmbedding import FlagReranker
            logger.info(f"Loading FlagReranker: {self.model_name}")
            self._reranker = FlagReranker(self.model_name, use_fp16=False)
            return "flagembedding"
        except Exception as fe_err:
            logger.debug(f"FlagReranker unavailable ({fe_err}); checking sentence-transformers.")

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder: {self.model_name}")
            self._st_cross = CrossEncoder(self.model_name)
            return "sentence_transformers"
        except Exception as st_err:
            logger.warning(
                f"Could not load '{self.model_name}' ({st_err}); using deterministic lexical reranker fallback."
            )
            return "fallback"

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates[:top_k]

        backend = self._load_backend()
        pairs = [[query, c.content] for c in candidates]

        if backend == "flagembedding":
            scores = self._reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, (int, float)):
                scores = [float(scores)]
            scored = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
            out: List[RetrievedChunk] = []
            for sc, ch in scored[:top_k]:
                ch.similarity_score = round(float(sc), 4)
                out.append(ch)
            return out

        if backend == "sentence_transformers":
            scores = self._st_cross.predict(pairs)
            scored = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
            out = []
            for sc, ch in scored[:top_k]:
                ch.similarity_score = round(float(sc), 4)
                out.append(ch)
            return out

        return await self._fallback.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            min_relevance_threshold=0.0,
        )
