import hashlib
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import logger
from app.core.observability import traceable
from app.documents.extractors.base import ExtractedChunk


class EmbeddedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    chunk_type: str
    extractor: str
    content: str
    content_hash: str
    heading_path: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    dense_embedding: List[float]
    sparse_values: Dict[int, float] = Field(default_factory=dict)
    extraction_meta: Dict[str, Any] = Field(default_factory=dict)


def _approx_token_count(text: str) -> int:
    # Approximate token count (~1.3 tokens per whitespace-delimited word)
    return int(len(text.split()) * 1.3)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


class AdaptiveChunker:
    """
    Post-extraction chunker enforcing content-type-specific chunking rules:
    - Text: target 512 tokens, 64-token overlap, hard cap 1024 tokens, sentence-boundary safe.
    - Table: 1 table = 1 chunk; if > 1024 tokens, split by rows repeating the header row
             and linking sub-chunks via parent_chunk_id.
    - Figure (figure_description / figure_caption): kept intact, never split.
    """

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        max_tokens: int = 1024,
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens

    def chunk_extracted(self, extracted: List[ExtractedChunk]) -> List[ExtractedChunk]:
        out: List[ExtractedChunk] = []
        for item in extracted:
            if item.chunk_type in ("figure_description", "figure_caption"):
                out.append(item)
            elif item.chunk_type == "table":
                out.extend(self._chunk_table(item))
            else:
                out.extend(self._chunk_text(item))
        return out

    def _chunk_text(self, item: ExtractedChunk) -> List[ExtractedChunk]:
        if _approx_token_count(item.content) <= self.target_tokens:
            return [item]

        sentences = _split_sentences(item.content)
        if not sentences:
            return [item]

        chunks: List[ExtractedChunk] = []
        current_sents: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _approx_token_count(sent)
            if current_sents and (current_tokens + sent_tokens > self.target_tokens or current_tokens + sent_tokens > self.max_tokens):
                chunk_text = " ".join(current_sents).strip()
                chunks.append(
                    ExtractedChunk(
                        doc_id=item.doc_id,
                        page_number=item.page_number,
                        bbox=item.bbox,
                        chunk_type="text",
                        content=chunk_text,
                        extractor=item.extractor,
                        heading_path=item.heading_path,
                        extraction_meta={**item.extraction_meta, "split": True},
                    )
                )
                # Build overlap from tail sentences
                overlap_sents: List[str] = []
                overlap_tok = 0
                for prev in reversed(current_sents):
                    ptok = _approx_token_count(prev)
                    if overlap_tok + ptok <= self.overlap_tokens:
                        overlap_sents.insert(0, prev)
                        overlap_tok += ptok
                    else:
                        break
                current_sents = overlap_sents
                current_tokens = overlap_tok

            current_sents.append(sent)
            current_tokens += sent_tokens

        if current_sents:
            chunk_text = " ".join(current_sents).strip()
            chunks.append(
                ExtractedChunk(
                    doc_id=item.doc_id,
                    page_number=item.page_number,
                    bbox=item.bbox,
                    chunk_type="text",
                    content=chunk_text,
                    extractor=item.extractor,
                    heading_path=item.heading_path,
                    extraction_meta={**item.extraction_meta, "split": len(chunks) > 0},
                )
            )

        return chunks

    def _chunk_table(self, item: ExtractedChunk) -> List[ExtractedChunk]:
        if _approx_token_count(item.content) <= self.max_tokens:
            return [item]

        lines = [ln for ln in item.content.splitlines() if ln.strip()]
        if len(lines) <= 2:
            return [item]

        # Markdown table header (title row + separator row if present)
        header_lines = lines[:2] if (len(lines) > 1 and set(lines[1].replace("|", "").replace("-", "").replace(":", "").strip()) == set()) else lines[:1]
        data_lines = lines[len(header_lines):]
        header_str = "\n".join(header_lines)
        header_tokens = _approx_token_count(header_str)

        parent_id = item.chunk_id
        sub_chunks: List[ExtractedChunk] = []
        current_rows: List[str] = []
        current_tokens = header_tokens

        for row in data_lines:
            row_tok = _approx_token_count(row)
            if current_rows and current_tokens + row_tok > self.max_tokens:
                content = header_str + "\n" + "\n".join(current_rows)
                sub_chunks.append(
                    ExtractedChunk(
                        doc_id=item.doc_id,
                        page_number=item.page_number,
                        bbox=item.bbox,
                        chunk_type="table",
                        content=content,
                        extractor=item.extractor,
                        heading_path=item.heading_path,
                        extraction_meta={
                            **item.extraction_meta,
                            "parent_chunk_id": parent_id,
                            "table_part": len(sub_chunks) + 1,
                        },
                    )
                )
                current_rows = []
                current_tokens = header_tokens

            current_rows.append(row)
            current_tokens += row_tok

        if current_rows:
            content = header_str + "\n" + "\n".join(current_rows)
            sub_chunks.append(
                ExtractedChunk(
                    doc_id=item.doc_id,
                    page_number=item.page_number,
                    bbox=item.bbox,
                    chunk_type="table",
                    content=content,
                    extractor=item.extractor,
                    heading_path=item.heading_path,
                    extraction_meta={
                        **item.extraction_meta,
                        "parent_chunk_id": parent_id,
                        "table_part": len(sub_chunks) + 1,
                    },
                )
            )

        return sub_chunks or [item]


class BgeM3Embedder:
    """
    Bi-encoder embedder using BAAI/bge-m3 to produce both:
    - dense_embedding: 1024-dim normalized float vector
    - sparse_values: {token_id: weight} lexical weights
    Defaults to CPU (with ONNX quantized support when configured) and batch_size=16.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimension: int = settings.EMBED_DIMENSION,
    ):
        self.model_name = model_name or settings.EMBED_MODEL
        self.batch_size = batch_size or settings.EMBED_BATCH_SIZE
        self.dimension = dimension
        self._flag_model = None
        self._st_model = None

    def _load_backend(self) -> str:
        if self._flag_model is not None:
            return "flagembedding"
        if self._st_model is not None:
            return "sentence_transformers"

        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading FlagEmbedding BGEM3FlagModel: {self.model_name}")
            self._flag_model = BGEM3FlagModel(self.model_name, use_fp16=False)
            return "flagembedding"
        except Exception as fe_err:
            logger.debug(f"FlagEmbedding unavailable ({fe_err}); checking sentence-transformers.")

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            backend = "onnx" if settings.EMBED_USE_ONNX else "torch"
            try:
                self._st_model = SentenceTransformer(self.model_name, backend=backend)
            except Exception:
                self._st_model = SentenceTransformer(self.model_name)
            return "sentence_transformers"
        except Exception as st_err:
            logger.warning(
                f"Neither FlagEmbedding nor SentenceTransformer could load '{self.model_name}' ({st_err}). "
                "Using deterministic lexical/hash bi-encoder fallback."
            )
            return "fallback"

    @staticmethod
    def _compute_lexical_sparse(text: str) -> Dict[int, float]:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return {}
        tf: Dict[int, float] = {}
        for tok in tokens:
            tid = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % 250000
            tf[tid] = tf.get(tid, 0.0) + 1.0
        max_tf = max(tf.values())
        return {tid: round(0.5 + 0.5 * (cnt / max_tf), 4) for tid, cnt in tf.items()}

    def _compute_deterministic_dense(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for tok in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    @traceable(name="bge_m3_encode_batch", run_type="embedding")
    def encode_batch(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        if not texts:
            return [], []

        backend = self._load_backend()

        if backend == "flagembedding":
            out = self._flag_model.encode(
                texts,
                batch_size=self.batch_size,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = [vec.tolist() for vec in out["dense_vecs"]]
            sparse = [
                {int(k): float(v) for k, v in lw.items()}
                for lw in out["lexical_weights"]
            ]
            return dense, sparse

        if backend == "sentence_transformers":
            dense_np = self._st_model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
            dense = [v.tolist() for v in dense_np]
            sparse = [self._compute_lexical_sparse(t) for t in texts]
            return dense, sparse

        dense = [self._compute_deterministic_dense(t) for t in texts]
        sparse = [self._compute_lexical_sparse(t) for t in texts]
        return dense, sparse

    @traceable(name="bge_m3_encode_query", run_type="embedding")
    def encode_query(self, query: str) -> Tuple[List[float], Dict[int, float]]:
        dense_list, sparse_list = self.encode_batch([query])
        return dense_list[0], sparse_list[0]

    def embed_extracted_chunks(self, chunks: List[ExtractedChunk]) -> List[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        dense_vecs, sparse_dicts = self.encode_batch(texts)

        embedded: List[EmbeddedChunk] = []
        for c, d_vec, s_dict in zip(chunks, dense_vecs, sparse_dicts):
            chash = hashlib.sha256(c.content.encode("utf-8")).hexdigest()
            parent_id = c.extraction_meta.get("parent_chunk_id") if c.extraction_meta else None
            embedded.append(
                EmbeddedChunk(
                    chunk_id=c.chunk_id or str(uuid.uuid4()),
                    doc_id=c.doc_id,
                    page_number=c.page_number,
                    bbox=c.bbox,
                    chunk_type=c.chunk_type,
                    extractor=c.extractor,
                    content=c.content,
                    content_hash=chash,
                    heading_path=c.heading_path,
                    parent_chunk_id=parent_id,
                    dense_embedding=d_vec,
                    sparse_values=s_dict,
                    extraction_meta=c.extraction_meta,
                )
            )
        return embedded
