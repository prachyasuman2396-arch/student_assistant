import asyncio
from typing import Any, Dict
from app.core.logging import logger
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks.dead_letter_handler", queue="dlq")
def dead_letter_handler(doc_id: str, user_id: str, error_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Dead-letter queue sink for permanently failed ingestion jobs."""
    logger.error(f"[DLQ] Ingestion permanently failed for doc={doc_id} user={user_id}: {error_payload}")
    return {"doc_id": doc_id, "user_id": user_id, "status": "dlq", "error": error_payload}


async def _run_async_ingestion(document_id: str, user_id: str) -> Dict[str, Any]:
    from app.db.session import AsyncSessionLocal
    from app.documents.ingestion_service import IngestionService

    async with AsyncSessionLocal() as session:
        service = IngestionService(session)
        doc = await service.run_ingestion_pipeline(document_id=document_id, user_id=user_id)
        return {
            "document_id": doc.id,
            "status": doc.status,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "triage_summary": doc.triage_summary,
        }


@celery_app.task(
    bind=True,
    name="app.workers.tasks.ingest_document_task",
    max_retries=3,
    default_retry_delay=5,
)
def ingest_document_task(self, document_id: str, user_id: str) -> Dict[str, Any]:
    """
    Celery ingestion entrypoint:
      triage -> extract -> chunk -> embed -> store -> status="ready"
    Retries 3x with exponential backoff, routing permanent failures to DLQ.
    """
    try:
        return asyncio.run(_run_async_ingestion(document_id=document_id, user_id=user_id))
    except Exception as exc:
        retry_num = self.request.retries
        if retry_num < self.max_retries:
            countdown = 5 * (2 ** retry_num)
            logger.warning(
                f"Retrying ingestion for doc={document_id} (attempt {retry_num + 1}/{self.max_retries}) in {countdown}s"
            )
            raise self.retry(exc=exc, countdown=countdown)

        dead_letter_handler.delay(
            document_id,
            user_id,
            {"error": str(exc), "error_type": type(exc).__name__},
        )
        raise
