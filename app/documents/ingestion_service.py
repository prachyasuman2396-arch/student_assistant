import asyncio
import hashlib
import os
import uuid
from typing import Any, Dict, List, Optional
import aiofiles
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.core.observability import traceable
from app.db.models.chunk import DocumentChunk
from app.db.models.document import Document
from app.db.repositories.document_repo import DocumentRepository
from app.documents.embedder import AdaptiveChunker, BgeM3Embedder
from app.documents.extractors import ExtractionCoordinator
from app.documents.parser import PDFParser
from app.documents.triage import PageTriager


class ObjectStorageClient:
    """
    Persists raw uploaded PDF bytes to S3/MinIO when configured,
    falling back transparently to the local upload volume.
    """

    @staticmethod
    async def save_bytes(storage_key: str, data: bytes) -> str:
        if settings.S3_ENDPOINT_URL and settings.S3_BUCKET:
            try:
                import httpx
                url = f"{settings.S3_ENDPOINT_URL.rstrip('/')}/{settings.S3_BUCKET}/{storage_key}"
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.put(url, content=data)
                    if resp.status_code in (200, 201, 204):
                        return url
                    logger.warning(f"S3/MinIO PUT returned {resp.status_code}; falling back to local disk.")
            except Exception as e:
                logger.warning(f"S3/MinIO storage failed ({e}); falling back to local disk.")

        os.makedirs(settings.UPLOAD_STORAGE_DIR, exist_ok=True)
        dest_path = os.path.join(settings.UPLOAD_STORAGE_DIR, storage_key)
        async with aiofiles.open(dest_path, "wb") as out_f:
            await out_f.write(data)
        return dest_path


class IngestionService:
    """
    Production async ingestion service:
      1. Fast upload endpoint (<500ms): validate, dedup via sha256(file_bytes)+user_id,
         persist raw bytes, create `documents` row with status="queued", enqueue Celery job.
      2. Worker pipeline: triage -> extract -> chunk -> embed -> store -> status="ready".
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = DocumentRepository(session)
        self.triager = PageTriager()
        self.extractor = ExtractionCoordinator()
        self.chunker = AdaptiveChunker()
        self.embedder = BgeM3Embedder()

    @staticmethod
    def compute_dedup_key(file_bytes: bytes, user_id: str) -> str:
        raw_hash = hashlib.sha256(file_bytes).hexdigest()
        return hashlib.sha256(f"{raw_hash}:{user_id}".encode("utf-8")).hexdigest()

    async def enqueue_upload(self, file: UploadFile, user_id: str) -> Document:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        file_bytes = await file.read()
        file_size = len(file_bytes)

        if file_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

        if not file_bytes[:5].startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Invalid PDF file signature or corrupted file.")

        dedup_hash = self.compute_dedup_key(file_bytes, user_id)
        existing_doc = await self.repo.find_by_content_hash(user_id=user_id, content_hash=dedup_hash)
        if existing_doc:
            logger.info(f"Idempotent upload hit for user={user_id}: returning existing doc {existing_doc.id}")
            return existing_doc

        storage_key = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
        stored_path = await ObjectStorageClient.save_bytes(storage_key, file_bytes)

        doc_record = await self.repo.create(
            user_id=user_id,
            filename=file.filename,
            file_path=stored_path,
            file_size_bytes=file_size,
            status="queued",
            content_hash=dedup_hash,
            mime_type="application/pdf",
            embedding_model=self.embedder.model_name,
        )

        # Dispatch to Celery queue (non-blocking)
        if settings.USE_ASYNC_INGESTION:
            try:
                from app.workers.tasks import ingest_document_task
                ingest_document_task.delay(doc_record.id, user_id)
                logger.info(f"Enqueued Celery ingestion job for doc={doc_record.id}")
            except Exception as celery_err:
                logger.warning(
                    f"Celery dispatch unavailable ({celery_err}); doc={doc_record.id} remains queued."
                )

        return doc_record

    @traceable(name="ingest_document_pipeline", run_type="chain")
    async def run_ingestion_pipeline(self, document_id: str, user_id: str) -> Document:
        """
        Full multi-extractor ingestion pipeline:
          triage -> extract -> chunk -> embed -> store -> status="ready"
        """
        doc = await self.repo.get_by_id(document_id=document_id, user_id=user_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found for user {user_id}")

        await self.repo.update_status(document_id=document_id, status="processing")
        stage = "triage"

        try:
            # 1. Page-level triage (whole document first)
            triage_res = self.triager.triage_pdf(doc.file_path)

            # 2. Routed extraction
            stage = "extract"
            extracted_chunks = await self.extractor.extract_document(
                file_path=doc.file_path,
                doc_id=doc.id,
                triage_result=triage_res,
            )

            # 3. Adaptive chunking
            stage = "chunk"
            chunked = self.chunker.chunk_extracted(extracted_chunks)

            # 4. Bi-encoder embedding (BAAI/bge-m3 dense + sparse)
            stage = "embed"
            embedded_chunks = self.embedder.embed_extracted_chunks(chunked)

            # 5. Store in pgvector
            stage = "store"
            orm_chunks: List[DocumentChunk] = []
            for idx, ec in enumerate(embedded_chunks):
                orm_chunks.append(
                    DocumentChunk(
                        id=ec.chunk_id,
                        document_id=doc.id,
                        user_id=user_id,
                        chunk_index=idx,
                        page_number=ec.page_number,
                        parent_chunk_id=ec.parent_chunk_id,
                        section_title=ec.heading_path,
                        page_start=ec.page_number,
                        page_end=ec.page_number,
                        bbox=list(ec.bbox) if ec.bbox else None,
                        chunk_type=ec.chunk_type,
                        extractor=ec.extractor,
                        content=ec.content,
                        content_hash=ec.content_hash,
                        dense_embedding=ec.dense_embedding,
                        sparse_values={str(k): v for k, v in ec.sparse_values.items()},
                        heading_path=ec.heading_path,
                        extraction_meta=ec.extraction_meta,
                        chunk_metadata={
                            "filename": doc.filename,
                            "page": ec.page_number,
                            "chunk_type": ec.chunk_type,
                            "extractor": ec.extractor,
                            "heading_path": ec.heading_path,
                        },
                    )
                )

            if orm_chunks:
                await self.repo.add_chunks(orm_chunks)

            updated = await self.repo.update_status(
                document_id=doc.id,
                status="ready",
                page_count=triage_res.total_pages,
                chunk_count=len(orm_chunks),
                triage_summary=triage_res.triage_summary,
            )
            logger.info(
                f"Document {doc.id} ready: {triage_res.total_pages} pages, "
                f"{len(orm_chunks)} chunks, triage={triage_res.triage_summary}"
            )
            return updated or doc

        except Exception as e:
            logger.error(f"Ingestion failed at stage '{stage}' for doc {document_id}: {e}", exc_info=True)
            await self.repo.update_status(
                document_id=document_id,
                status="failed",
                error_message=str(e),
                last_error={"stage": stage, "error": str(e), "error_type": type(e).__name__},
            )
            raise
