import json
import re
from typing import Dict, Any, List, Optional
from app.graph.state import ExecutionPlan, Task, TaskStatus
from app.graph.validator import REGISTERED_WORKERS
from app.llm.router import LLMRouter
from app.llm.schemas import TaskType, ComplexityLevel, LLMMessage
from app.core.logging import logger


PLANNER_SYSTEM_PROMPT = """You are an expert AI Study Architect and Planner for a student learning system.
Your job is to analyze the student's query and decompose it into an optimal, dependency-aware Directed Acyclic Graph (DAG) of executable tasks.

AVAILABLE WORKERS:
- "rag": Retrieve excerpts from uploaded textbooks/PDFs using vector similarity. Use when the student explicitly mentions "uploaded document", "textbook", "PDF", "chapter", "lecture notes", or when document_id is provided.
- "research": Query PubMed, Europe PMC, and biomedical databases for peer-reviewed papers, clinical trials, or systematic reviews. Use when query asks for research, recent papers, trials, or evidence.
- "tutor": Synthesizes educational explanations, mechanism breakdowns, or compares upstream RAG and research evidence.
- "mcq": Generates high-yield multiple-choice examination questions with options, correct answers, and explanations.
- "flashcard": Generates active-recall question-and-answer study flashcards.
- "direct_answer": Fast, single-turn answer for simple greetings or trivial questions.

DIRECT EXECUTION RULE:
For simple, single-intent questions (e.g., "What is reversible pulpitis?", "Hello", "Define osteoblast"), DO NOT overcomplicate with multiple tasks. Return a single task or mark is_direct=true.

MULTI-TASK DECOMPOSITION RULE:
For compound queries, decompose into clean independent stages.
Example: "Explain irreversible pulpitis using my uploaded textbook, compare it with recent research papers and generate five MCQs."
Tasks:
- T1: worker="rag", dependencies=[], description="Retrieve textbook passages on irreversible pulpitis"
- T2: worker="research", dependencies=[], description="Retrieve recent PubMed research on irreversible pulpitis"
- T3: worker="tutor", dependencies=["T1", "T2"], description="Compare textbook content and research findings into a comprehensive synthesis"
- T4: worker="mcq", dependencies=["T3"], description="Generate 5 MCQs based on the synthesis"

OUTPUT SCHEMA:
Respond STRICTLY with a valid JSON object matching this structure:
{
  "objective": "Brief summary of what will be achieved",
  "is_direct": false,
  "final_response_strategy": "synthesize",
  "tasks": [
    {
      "task_id": "T1",
      "worker": "rag" | "research" | "tutor" | "mcq" | "flashcard" | "direct_answer",
      "description": "Clear description of what this worker does",
      "dependencies": ["T0"],
      "expected_output": "What this task produces",
      "input_mapping": {"query": "search query if applicable"}
    }
  ]
}

Ensure:
- task_ids are unique (e.g. T1, T2, T3)
- dependencies only reference prior task_ids
- NO cyclic dependencies
- only use valid worker names
"""


class PlannerAgent:
    """
    Dedicated Planner Agent responsible for dynamic query decomposition into an ExecutionPlan DAG.
    """

    def __init__(self, router: LLMRouter):
        self.router = router

    async def generate_plan(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        document_id: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        revision_feedback: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Generates an ExecutionPlan for the user query, handling direct shortcuts,
        structured LLM decomposition, and revision feedback.
        """
        clean_query = query.strip()
        low_query = clean_query.lower()

        # Fast heuristic check for simple single-intent queries
        if not revision_feedback and not document_id:
            clean_low = re.sub(r"[^\w\s]", "", low_query).strip()
            # Simple greetings or trivial questions
            if (
                clean_low in ("hi", "hello", "hey", "hello there", "hey there", "help", "who are you")
                or clean_low.startswith(("hello ", "hey ", "hi "))
            ):
                return ExecutionPlan(
                    objective="Acknowledge student greeting",
                    is_direct=True,
                    final_response_strategy="direct",
                    tasks=[
                        Task(
                            task_id="T1",
                            worker="direct_answer",
                            description="Respond warmly to student greeting and offer study assistance",
                            dependencies=[],
                            expected_output="Greeting response",
                            input_mapping={"query": clean_query},
                            status=TaskStatus.PENDING,
                        )
                    ],
                )

        # Build context
        context_parts = [f"Student Query: {clean_query}"]
        if document_id:
            context_parts.append(f"Uploaded Document ID available: {document_id}")
        if profile:
            context_parts.append(
                f"Student Level: {profile.get('current_level', 'Undergraduate')}, "
                f"Discipline: {profile.get('academic_discipline', 'General')}"
            )
        if revision_feedback:
            context_parts.append(
                f"\nCRITICAL: Your previous plan was REJECTED with error: {revision_feedback}\n"
                f"Fix the error and produce a strictly valid DAG plan."
            )

        messages = [
            LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
            LLMMessage(role="user", content="\n".join(context_parts)),
        ]

        try:
            res = await self.router.execute_with_fallback(
                task_type=TaskType.CLASSIFICATION,
                messages=messages,
                complexity=ComplexityLevel.MEDIUM if revision_feedback else ComplexityLevel.LOW,
                temperature=0.2,
                max_tokens=600,
            )

            raw = res.content.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            plan_dict = json.loads(raw)
            return self._parse_and_sanitize_plan(plan_dict, clean_query, document_id)

        except Exception as e:
            logger.warning(f"Planner LLM generation or JSON parse failed: {e}. Falling back to deterministic plan.")
            return self._build_heuristic_plan(clean_query, document_id)

    def _parse_and_sanitize_plan(
        self,
        data: Dict[str, Any],
        fallback_query: str,
        document_id: Optional[str] = None,
    ) -> ExecutionPlan:
        objective = data.get("objective") or f"Address: {fallback_query[:50]}"
        is_direct = bool(data.get("is_direct", False))
        strategy = data.get("final_response_strategy", "synthesize")
        raw_tasks = data.get("tasks", [])

        tasks: List[Task] = []
        for item in raw_tasks:
            worker_name = str(item.get("worker", "tutor")).lower().strip()
            if worker_name not in REGISTERED_WORKERS:
                worker_name = "tutor"

            input_map = item.get("input_mapping") or {}
            if not isinstance(input_map, dict):
                input_map = {}
            if "query" not in input_map:
                input_map["query"] = fallback_query
            if document_id and "document_id" not in input_map:
                input_map["document_id"] = document_id

            t = Task(
                task_id=str(item.get("task_id", f"T{len(tasks)+1}")),
                worker=worker_name,
                description=str(item.get("description", "Execute task")),
                dependencies=[str(d) for d in item.get("dependencies", [])],
                expected_output=str(item.get("expected_output", "Task result")),
                input_mapping=input_map,
                status=TaskStatus.PENDING,
            )
            tasks.append(t)

        if not tasks:
            return self._build_heuristic_plan(fallback_query, document_id)

        return ExecutionPlan(
            objective=objective,
            is_direct=is_direct if len(tasks) == 1 else False,
            final_response_strategy=strategy,
            tasks=tasks,
        )

    def _build_heuristic_plan(
        self,
        query: str,
        document_id: Optional[str] = None,
    ) -> ExecutionPlan:
        """Deterministic rule-based fallback plan for reliability when LLM plan generation fails."""
        low = query.lower()
        has_doc = bool(document_id) or any(w in low for w in ["document", "pdf", "textbook", "page", "slide"])
        has_research = any(w in low for w in ["research", "pubmed", "paper", "trial", "study", "guideline"])
        has_mcq = any(w in low for w in ["mcq", "quiz", "multiple choice", "test questions"])
        has_flashcard = any(w in low for w in ["flashcard", "card", "active recall", "deck"])

        tasks: List[Task] = []

        if has_doc and has_research:
            tasks.append(Task(
                task_id="T1",
                worker="rag",
                description="Retrieve relevant textbook/PDF passages",
                dependencies=[],
                expected_output="Document chunks",
                input_mapping={"query": query, "document_id": document_id},
            ))
            tasks.append(Task(
                task_id="T2",
                worker="research",
                description="Retrieve recent scientific literature",
                dependencies=[],
                expected_output="Biomedical research articles",
                input_mapping={"query": query},
            ))
            tasks.append(Task(
                task_id="T3",
                worker="tutor",
                description="Compare and synthesize textbook and research findings",
                dependencies=["T1", "T2"],
                expected_output="Comprehensive synthesis",
                input_mapping={"query": query},
            ))
            if has_mcq:
                tasks.append(Task(
                    task_id="T4",
                    worker="mcq",
                    description="Generate MCQs on synthesized topic",
                    dependencies=["T3"],
                    expected_output="Multiple choice questions",
                    input_mapping={"query": query},
                ))
            elif has_flashcard:
                tasks.append(Task(
                    task_id="T4",
                    worker="flashcard",
                    description="Generate flashcards on synthesized topic",
                    dependencies=["T3"],
                    expected_output="Study flashcards",
                    input_mapping={"query": query},
                ))
            return ExecutionPlan(
                objective=f"Comparative analysis: {query[:50]}",
                is_direct=False,
                tasks=tasks,
                final_response_strategy="synthesize",
            )

        if has_doc:
            tasks.append(Task(
                task_id="T1",
                worker="rag",
                description="Retrieve relevant textbook/PDF passages",
                dependencies=[],
                expected_output="Document chunks",
                input_mapping={"query": query, "document_id": document_id},
            ))
            tasks.append(Task(
                task_id="T2",
                worker="tutor",
                description="Synthesize educational response grounded in document passages",
                dependencies=["T1"],
                expected_output="Document-grounded answer",
                input_mapping={"query": query},
            ))
            if has_mcq:
                tasks.append(Task(
                    task_id="T3",
                    worker="mcq",
                    description="Generate MCQs",
                    dependencies=["T2"],
                    expected_output="MCQs",
                    input_mapping={"query": query},
                ))
            return ExecutionPlan(
                objective=f"Document-grounded query: {query[:50]}",
                is_direct=False,
                tasks=tasks,
                final_response_strategy="synthesize",
            )

        if has_research:
            tasks.append(Task(
                task_id="T1",
                worker="research",
                description="Search PubMed and scholarly literature",
                dependencies=[],
                expected_output="Research papers",
                input_mapping={"query": query},
            ))
            tasks.append(Task(
                task_id="T2",
                worker="tutor",
                description="Synthesize evidence-based response",
                dependencies=["T1"],
                expected_output="Evidence synthesis",
                input_mapping={"query": query},
            ))
            return ExecutionPlan(
                objective=f"Research synthesis: {query[:50]}",
                is_direct=False,
                tasks=tasks,
                final_response_strategy="synthesize",
            )

        if has_mcq:
            tasks.append(Task(
                task_id="T1",
                worker="mcq",
                description="Generate practice MCQs",
                dependencies=[],
                expected_output="MCQ questions",
                input_mapping={"query": query},
            ))
            return ExecutionPlan(
                objective=f"Generate MCQs: {query[:50]}",
                is_direct=False,
                tasks=tasks,
                final_response_strategy="direct",
            )

        if has_flashcard:
            tasks.append(Task(
                task_id="T1",
                worker="flashcard",
                description="Generate study flashcards",
                dependencies=[],
                expected_output="Flashcards",
                input_mapping={"query": query},
            ))
            return ExecutionPlan(
                objective=f"Generate flashcards: {query[:50]}",
                is_direct=False,
                tasks=tasks,
                final_response_strategy="direct",
            )

        # Default single direct tutor task
        tasks.append(Task(
            task_id="T1",
            worker="tutor",
            description="Provide comprehensive pedagogical explanation",
            dependencies=[],
            expected_output="Educational explanation",
            input_mapping={"query": query},
        ))
        return ExecutionPlan(
            objective=f"Study explanation: {query[:50]}",
            is_direct=True,
            tasks=tasks,
            final_response_strategy="direct",
        )
