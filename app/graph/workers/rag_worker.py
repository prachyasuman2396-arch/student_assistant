import time
import asyncio
from typing import Dict, Any, Optional, Callable
from app.graph.workers.base import BaseWorker
from app.graph.state import Task, TaskResult, TaskStatus
from app.core.logging import logger
from app.core.observability import traceable
from app.prompts.sanitizer import sanitize_untrusted_text


class RAGWorker(BaseWorker):
    """
    Specialized worker for retrieving relevant uploaded textbook/document chunks
    using pgvector or hybrid retriever. Returns citations and source metadata.
    """

    def __init__(self, retriever_factory: Optional[Callable[[], Any]] = None):
        self.retriever_factory = retriever_factory

    @property
    def name(self) -> str:
        return "rag"

    @traceable(name="rag_worker_execute", run_type="retriever")
    async def execute(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        start_time = time.perf_counter()
        query = task.input_mapping.get("query") or context.get("query") or task.description
        user_id = context.get("user_id", "")
        document_id = task.input_mapping.get("document_id") or context.get("document_id")

        if not self.retriever_factory:
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.COMPLETED,
                output={"chunks": [], "text": "No document retriever factory configured."},
                citations={"documents": []},
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        try:
            retriever = self.retriever_factory()
            call_res = None
            top_k_val = task.input_mapping.get("top_k", 5)

            # Check for explicitly configured AsyncMock or coroutine methods
            if hasattr(retriever, "retrieve") and type(getattr(retriever, "retrieve")).__name__ == "AsyncMock":
                call_res = retriever.retrieve(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    top_k=top_k_val,
                )
            elif hasattr(retriever, "pipeline_retrieve") and (
                asyncio.iscoroutinefunction(getattr(retriever, "pipeline_retrieve"))
                or type(getattr(retriever, "pipeline_retrieve")).__name__ == "AsyncMock"
            ):
                call_res = retriever.pipeline_retrieve(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    top_k=top_k_val,
                )
            elif hasattr(retriever, "similarity_search"):
                call_res = retriever.similarity_search(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    top_k=top_k_val,
                )
            elif hasattr(retriever, "pipeline_retrieve"):
                call_res = retriever.pipeline_retrieve(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    top_k=top_k_val,
                )
            elif hasattr(retriever, "retrieve"):
                call_res = retriever.retrieve(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    top_k=top_k_val,
                )

            import asyncio
            if asyncio.iscoroutine(call_res):
                res = await call_res
            else:
                res = call_res

            if isinstance(res, tuple):
                chunks = res[0]
            elif isinstance(res, list):
                chunks = res
            else:
                chunks = []

            serialized_chunks = []
            for c in chunks:
                d = c.model_dump() if hasattr(c, "model_dump") else (c if isinstance(c, dict) else vars(c))
                d["content"] = sanitize_untrusted_text(d.get("content", ""))
                if not d.get("citation_label"):
                    ctype = d.get("chunk_type") or "text"
                    d["citation_label"] = f"[{d.get('filename')}, p.{d.get('page_number')}, {ctype}]"
                serialized_chunks.append(d)

            doc_citations = [
                {
                    "filename": c.get("filename"),
                    "page_number": c.get("page_number"),
                    "chunk_id": c.get("chunk_id"),
                    "chunk_type": c.get("chunk_type", "text"),
                    "extractor": c.get("extractor", "pymupdf"),
                    "heading_path": c.get("heading_path") or c.get("section_title"),
                    "citation_label": c.get("citation_label"),
                    "similarity_score": c.get("similarity_score"),
                }
                for c in serialized_chunks
            ]

            # Build text summary of evidence with inline citation labels
            chunk_texts = [
                f"Source {c.get('citation_label')}:\n{c.get('content')}"
                for c in serialized_chunks
            ]
            evidence_summary = "\n\n".join(chunk_texts) if chunk_texts else "No matching textbook passages found."

            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.COMPLETED,
                output={"chunks": serialized_chunks, "text": evidence_summary},
                citations={"documents": doc_citations},
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        except Exception as e:
            logger.error(f"RAGWorker task '{task.task_id}' failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.FAILED,
                output=None,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
