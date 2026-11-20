import time
from typing import Dict, Any, Optional
from app.graph.workers.base import BaseWorker
from app.graph.state import Task, TaskResult, TaskStatus
from app.research.service import ResearchService
from app.core.logging import logger


class ResearchWorker(BaseWorker):
    """
    Specialized worker querying PubMed, Europe PMC, and biomedical databases.
    Returns structured articles, citations, and evidence-level classifications.
    """

    def __init__(self, research_service: Optional[ResearchService] = None):
        self.research = research_service or ResearchService()

    @property
    def name(self) -> str:
        return "research"

    async def execute(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        start_time = time.perf_counter()
        query = task.input_mapping.get("query") or context.get("query") or task.description
        max_results = task.input_mapping.get("max_results", 5)

        try:
            articles = await self.research.search_literature(query=query, max_results=max_results)
            serialized_articles = [
                a.model_dump() if hasattr(a, "model_dump") else (a if isinstance(a, dict) else vars(a))
                for a in articles
            ]

            paper_citations = [
                {
                    "title": a.get("title"),
                    "authors": a.get("authors"),
                    "journal": a.get("journal"),
                    "pub_date": a.get("publication_date"),
                    "pmid": a.get("pmid"),
                    "doi": a.get("doi"),
                    "url": a.get("url"),
                    "evidence_level": a.get("evidence_level"),
                }
                for a in serialized_articles
            ]

            lines = []
            for a in serialized_articles:
                lines.append(
                    f"Paper: {a.get('title')}\n"
                    f"Authors: {a.get('authors')}\n"
                    f"Journal: {a.get('journal')} ({a.get('publication_date')})\n"
                    f"Evidence Level: {a.get('evidence_level')}\n"
                    f"Abstract/Summary: {a.get('abstract') or 'No abstract available.'}"
                )
            evidence_summary = "\n\n".join(lines) if lines else "No peer-reviewed papers found for this query."

            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.COMPLETED,
                output={"articles": serialized_articles, "text": evidence_summary},
                citations={"research_papers": paper_citations},
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        except Exception as e:
            logger.error(f"ResearchWorker task '{task.task_id}' failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.FAILED,
                output=None,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
