import time
from typing import Dict, Any, Optional, List
from app.graph.workers.base import BaseWorker
from app.graph.state import Task, TaskResult, TaskStatus
from app.llm.router import LLMRouter
from app.llm.schemas import TaskType, ComplexityLevel, LLMMessage
from app.prompts.tutor import DENTAL_TUTOR_SYSTEM_PROMPT
from app.prompts.research import RESEARCH_SYNTHESIS_SYSTEM_PROMPT
from app.core.logging import logger


class TutorWorker(BaseWorker):
    """
    Specialized pedagogical tutor worker.
    Capable of conceptual explanations, evidence synthesis across upstream RAG and Research workers,
    and adapting tone to student academic profile and memory.
    """

    def __init__(self, router: LLMRouter):
        self.router = router

    @property
    def name(self) -> str:
        return "tutor"

    async def execute(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        start_time = time.perf_counter()
        query = task.input_mapping.get("query") or context.get("query") or task.description
        upstream_results = context.get("upstream_results", {})
        profile = context.get("profile") or {}
        semantic_memories = context.get("semantic_memories") or []
        episodic_memories = context.get("episodic_memories") or []

        # Gather upstream evidence if available
        evidence_sections: List[str] = []
        merged_citations: Dict[str, List[Any]] = {"documents": [], "research_papers": []}

        for dep_id in task.dependencies:
            dep_res = upstream_results.get(dep_id)
            if not dep_res or not dep_res.output:
                continue

            if isinstance(dep_res.output, dict):
                text = dep_res.output.get("text", "")
                if text:
                    evidence_sections.append(f"=== EVIDENCE FROM TASK {dep_id} ({dep_res.worker.upper()}) ===\n{text}")

            if dep_res.citations:
                if "documents" in dep_res.citations:
                    merged_citations["documents"].extend(dep_res.citations["documents"])
                if "research_papers" in dep_res.citations:
                    merged_citations["research_papers"].extend(dep_res.citations["research_papers"])

        # Decide whether to use synthesis prompt or tutor prompt
        has_evidence = len(evidence_sections) > 0
        system_base = RESEARCH_SYNTHESIS_SYSTEM_PROMPT if has_evidence else DENTAL_TUTOR_SYSTEM_PROMPT

        # Append student personalization
        personalization = []
        if profile.get("preferred_name"):
            personalization.append(f"- Student Name: {profile['preferred_name']}")
        if profile.get("academic_discipline"):
            personalization.append(f"- Discipline: {profile['academic_discipline']}")
        if profile.get("current_level"):
            personalization.append(f"- Study Level: {profile['current_level']}")
        if profile.get("explanation_style"):
            personalization.append(f"- Explanation Style: {profile['explanation_style']}")
        if profile.get("learning_goals"):
            personalization.append(f"- Target Learning Goals: {', '.join(profile['learning_goals'])}")

        mem_parts = [f"- {m.get('topic') or 'Context'}: {m.get('content')}" for m in semantic_memories[:3]]
        episode_parts = [f"- Past Event ({e.get('event_type')}): {e.get('summary')}" for e in episodic_memories[:2]]

        custom_context = ""
        if personalization:
            custom_context += "\n\n### STUDENT PROFILE & PREFERENCES\n" + "\n".join(personalization)
        if mem_parts:
            custom_context += "\n\n### RELEVANT LONG-TERM STUDY PREFERENCES\n" + "\n".join(mem_parts)
        if episode_parts:
            custom_context += "\n\n### RELEVANT PAST LEARNING EPISODES\n" + "\n".join(episode_parts)

        # Build prompt messages
        system_instruction = system_base + custom_context

        if has_evidence:
            prompt_content = (
                f"Student Question: {query}\n\n"
                f"TASK GOAL: {task.description}\n\n"
                f"EVIDENCE DOSSIER:\n" + "\n\n".join(evidence_sections) + "\n\n"
                f"Provide a comprehensive, pedagogical, evidence-grounded answer based strictly on the above dossier."
            )
        else:
            prompt_content = f"Student Question: {query}\n\nTASK GOAL: {task.description}"

        messages = [
            LLMMessage(role="system", content=system_instruction),
            LLMMessage(role="user", content=prompt_content),
        ]

        try:
            res = await self.router.execute_with_fallback(
                task_type=TaskType.RESEARCH_SYNTHESIS if has_evidence else TaskType.TUTOR,
                messages=messages,
                complexity=ComplexityLevel.HIGH if has_evidence else ComplexityLevel.MEDIUM,
                temperature=0.5,
            )

            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.COMPLETED,
                output={"text": res.content, "provider": res.provider.value, "model": res.model},
                citations=merged_citations,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        except Exception as e:
            logger.error(f"TutorWorker task '{task.task_id}' failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.FAILED,
                output=None,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
