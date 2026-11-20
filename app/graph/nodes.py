import json
import re
import time
from typing import Dict, Any, List, Optional, Callable
from app.graph.state import (
    AgentState,
    ExecutionPlan,
    Task,
    TaskResult,
    TaskStatus,
)
from app.graph.planner import PlannerAgent
from app.graph.validator import PlanValidator
from app.graph.orchestrator import Orchestrator
from app.graph.aggregator import ResultAggregator
from app.graph.workers import (
    BaseWorker,
    RAGWorker,
    ResearchWorker,
    TutorWorker,
    MCQWorker,
    FlashcardWorker,
    DirectAnswerWorker,
)
from app.memory.service import MemoryService
from app.llm.router import LLMRouter
from app.llm.schemas import TaskType, ComplexityLevel, LLMMessage
from app.prompts.tutor import DENTAL_TUTOR_SYSTEM_PROMPT
from app.prompts.classifier import CLASSIFICATION_SYSTEM_PROMPT
from app.prompts.research import RESEARCH_SYNTHESIS_SYSTEM_PROMPT
from app.prompts.study import QUIZ_GENERATION_PROMPT, FLASHCARD_GENERATION_PROMPT
from app.research.service import ResearchService
from app.core.logging import logger

MAX_PLAN_REVISIONS = 2


class DentalGraphNodes:
    """
    Multi-agent graph nodes supporting the complete
    Memory Retrieval -> Planner -> Plan Validator -> Orchestrator -> Result Aggregator -> Memory Update
    pipeline, while preserving legacy node methods for backward compatibility.
    """

    def __init__(
        self,
        router: LLMRouter,
        research_service: Optional[ResearchService] = None,
        retriever_factory: Optional[Callable[[], Any]] = None,
        profile_repo_factory: Optional[Callable[[], Any]] = None,
        memory_repo_factory: Optional[Callable[[], Any]] = None,
        episodic_repo_factory: Optional[Callable[[], Any]] = None,
        study_service_factory: Optional[Callable[[], Any]] = None,
        memory_service: Optional[MemoryService] = None,
    ):
        self.router = router
        self.research = research_service or ResearchService()
        self.retriever_factory = retriever_factory
        self.profile_repo_factory = profile_repo_factory
        self.memory_repo_factory = memory_repo_factory
        self.episodic_repo_factory = episodic_repo_factory
        self.study_service_factory = study_service_factory

        # Initialize core components
        self.planner = PlannerAgent(router=self.router)
        self.validator = PlanValidator()
        self.aggregator = ResultAggregator()

        # Initialize workers
        self.workers: Dict[str, BaseWorker] = {
            "rag": RAGWorker(retriever_factory=self.retriever_factory),
            "research": ResearchWorker(research_service=self.research),
            "tutor": TutorWorker(router=self.router),
            "mcq": MCQWorker(router=self.router, study_service_factory=self.study_service_factory),
            "flashcard": FlashcardWorker(router=self.router, study_service_factory=self.study_service_factory),
            "direct_answer": DirectAnswerWorker(router=self.router),
        }

        self.orchestrator = Orchestrator(workers=self.workers)
        self.memory_service = memory_service

    def _get_memory_service(self) -> MemoryService:
        if self.memory_service:
            return self.memory_service
        p_repo = self.profile_repo_factory() if self.profile_repo_factory else None
        m_repo = self.memory_repo_factory() if self.memory_repo_factory else None
        e_repo = self.episodic_repo_factory() if self.episodic_repo_factory else None
        return MemoryService(
            router=self.router,
            profile_repo=p_repo,
            memory_repo=m_repo,
            episodic_repo=e_repo,
        )

    # =========================================================================
    # PHASE 1: Memory Retrieval Node
    # =========================================================================
    async def retrieve_memories(self, state: AgentState) -> Dict[str, Any]:
        """Loads short-term trimmed context, student profile, semantic memories, and past episodes."""
        messages = state.get("messages", [])
        user_id = state.get("user_id", "")
        last_query = messages[-1].get("content", "") if messages else ""

        mem_svc = self._get_memory_service()
        trimmed_messages = mem_svc.trim_messages(messages, max_messages=12)
        memory_bundle = await mem_svc.retrieve_memories_for_query(user_id=user_id, query=last_query)

        profile_data = state.get("profile") or memory_bundle.get("profile")
        semantic_mems = memory_bundle.get("semantic_memories") or []
        episodic_mems = memory_bundle.get("episodic_memories") or []

        return {
            "messages": trimmed_messages,
            "profile": profile_data,
            "semantic_memories": semantic_mems,
            "long_term_memories": semantic_mems,  # Legacy alias
            "episodic_memories": episodic_mems,
            "plan_revision_count": 0,
            "plan_is_valid": False,
            "plan_validation_error": None,
            "retry_count": state.get("retry_count", 0),
            "citations": state.get("citations", {}),
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "research_sources": state.get("research_sources", []),
            "evidence_sufficient": True,
        }

    # Backward compatibility alias
    load_context = retrieve_memories

    # =========================================================================
    # PHASE 2: Planner Agent Node
    # =========================================================================
    async def plan_query(self, state: AgentState) -> Dict[str, Any]:
        """Analyzes query and dynamically decomposes it into an ExecutionPlan DAG."""
        messages = state.get("messages", [])
        last_query = messages[-1].get("content", "") if messages else ""
        document_id = state.get("document_id")
        profile = state.get("profile")
        revision_feedback = state.get("plan_validation_error")

        plan = await self.planner.generate_plan(
            query=last_query,
            conversation_history=messages,
            document_id=document_id,
            profile=profile,
            revision_feedback=revision_feedback,
        )

        # Determine backward-compatible intent
        intent = "tutor"
        if plan.tasks:
            workers = [t.worker for t in plan.tasks]
            if "rag" in workers and "research" in workers:
                intent = "hybrid_retrieval"
            elif "rag" in workers:
                intent = "document_retrieval"
            elif "research" in workers:
                intent = "web_research"
            elif "mcq" in workers or "flashcard" in workers:
                intent = "study_tools"
            else:
                intent = "tutor"

        return {
            "plan": plan.model_dump(),
            "intent": intent,
            "task_complexity": "low" if plan.is_direct else "medium",
        }

    # =========================================================================
    # PHASE 3: Plan Validator Node
    # =========================================================================
    async def validate_plan(self, state: AgentState) -> Dict[str, Any]:
        """Validates DAG structure, unique task IDs, registered workers, and cycle absence."""
        plan_raw = state.get("plan")
        revision_count = state.get("plan_revision_count", 0)

        if not plan_raw:
            return {
                "plan_is_valid": False,
                "plan_validation_error": "Plan is missing from state.",
                "plan_revision_count": revision_count + 1,
            }

        try:
            plan = ExecutionPlan.model_validate(plan_raw)
            is_valid, error = self.validator.validate(plan)
        except Exception as e:
            is_valid, error = False, str(e)

        if is_valid:
            return {
                "plan_is_valid": True,
                "plan_validation_error": None,
            }
        else:
            logger.warning(f"Plan validation rejected (attempt {revision_count+1}): {error}")
            return {
                "plan_is_valid": False,
                "plan_validation_error": error,
                "plan_revision_count": revision_count + 1,
            }

    # =========================================================================
    # PHASE 4: Deterministic Orchestrator Node
    # =========================================================================
    async def orchestrate_tasks(self, state: AgentState) -> Dict[str, Any]:
        """Executes tasks according to validated DAG, dispatching ready tasks concurrently."""
        plan_raw = state.get("plan")
        messages = state.get("messages", [])
        last_query = messages[-1].get("content", "") if messages else ""
        user_id = state.get("user_id", "")
        document_id = state.get("document_id")

        if not plan_raw:
            # Fallback direct execution
            plan = self.planner._build_heuristic_plan(last_query, document_id)
        else:
            plan = ExecutionPlan.model_validate(plan_raw)

        context = {
            "query": last_query,
            "user_id": user_id,
            "conversation_id": state.get("conversation_id", ""),
            "document_id": document_id,
            "profile": state.get("profile"),
            "semantic_memories": state.get("semantic_memories", []),
            "episodic_memories": state.get("episodic_memories", []),
        }

        results = await self.orchestrator.execute_plan(plan, context)

        serialized_results = [r.model_dump() for r in results]
        completed_ids = [r.task_id for r in results if r.status == TaskStatus.COMPLETED]
        failed_ids = [r.task_id for r in results if r.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)]

        return {
            "task_results": serialized_results,
            "completed_task_ids": completed_ids,
            "failed_task_ids": failed_ids,
        }

    # =========================================================================
    # PHASE 5: Result Aggregator Node
    # =========================================================================
    async def aggregate_results(self, state: AgentState) -> Dict[str, Any]:
        """Synthesizes final response, merges citations, and compiles study artifacts."""
        task_results_raw = state.get("task_results", [])
        plan_raw = state.get("plan")
        messages = state.get("messages", [])
        last_query = messages[-1].get("content", "") if messages else ""

        plan = ExecutionPlan.model_validate(plan_raw) if plan_raw else None
        results = [TaskResult.model_validate(r) for r in task_results_raw]

        aggregation = self.aggregator.aggregate(
            task_results=results,
            plan=plan,
            original_query=last_query,
        )

        return {
            "final_answer": aggregation["final_answer"],
            "answer": aggregation["answer"],  # Legacy compatibility
            "citations": aggregation["citations"],
            "study_artifacts": aggregation["study_artifacts"],
            "retrieved_chunks": aggregation["retrieved_chunks"],
            "research_sources": aggregation["research_sources"],
            "selected_provider": "groq",
            "selected_model": "multi-agent-orchestrated",
        }

    # =========================================================================
    # PHASE 6: Memory Update Node
    # =========================================================================
    async def update_memories(self, state: AgentState) -> Dict[str, Any]:
        """Extracts and persists semantic preferences and episodic events."""
        user_id = state.get("user_id", "")
        messages = state.get("messages", [])
        last_query = messages[-1].get("content", "") if messages else ""
        final_answer = state.get("final_answer") or state.get("answer", "")
        artifacts = state.get("study_artifacts", [])
        session_id = state.get("conversation_id")

        mem_svc = self._get_memory_service()
        update_result = await mem_svc.extract_and_update_memories(
            user_id=user_id,
            query=last_query,
            final_answer=final_answer,
            study_artifacts=artifacts,
            session_id=session_id,
        )

        return {
            "extracted_memories": update_result.get("semantic", []) + update_result.get("episodic", []),
        }

    # Backward compatibility alias
    save_turn = update_memories

    # =========================================================================
    # LEGACY GRAPH NODE METHODS (Preserved 100% for existing unit tests)
    # =========================================================================
    async def classify_query(self, state: AgentState) -> Dict[str, Any]:
        """Legacy classifier preserved for existing test_graph_routing.py."""
        messages = state.get("messages", [])
        if not messages:
            return {"intent": "tutor", "task_complexity": "low"}

        last_user_msg = messages[-1].get("content", "")
        document_id = state.get("document_id")

        if document_id and any(w in last_user_msg.lower() for w in ["document", "pdf", "textbook", "page", "slide", "chapter"]):
            return {
                "intent": "document_retrieval",
                "task_complexity": "medium",
                "search_query": last_user_msg,
            }

        prompt_msgs = [
            LLMMessage(role="system", content=CLASSIFICATION_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"User Query: {last_user_msg}"),
        ]

        try:
            res = await self.router.execute_with_fallback(
                task_type=TaskType.CLASSIFICATION,
                messages=prompt_msgs,
                complexity=ComplexityLevel.LOW,
                temperature=0.1,
                max_tokens=250,
            )
            raw = res.content.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            intent = parsed.get("intent", "tutor")
            complexity = parsed.get("complexity", "low")
            search_query = parsed.get("search_query") or last_user_msg

            if document_id and intent == "tutor":
                intent = "document_retrieval"

            return {
                "intent": intent,
                "task_complexity": complexity,
                "search_query": search_query,
                "selected_provider": res.provider.value,
                "selected_model": res.model,
            }
        except Exception:
            low = last_user_msg.lower()
            intent = "tutor"
            if any(k in low for k in ["mcq", "quiz", "flashcard", "viva"]):
                intent = "study_tools"
            elif any(k in low for k in ["research", "pubmed", "trial"]):
                intent = "web_research"
            elif document_id:
                intent = "document_retrieval"
            return {"intent": intent, "task_complexity": "low", "search_query": last_user_msg}

    async def tutor(self, state: AgentState) -> Dict[str, Any]:
        """Legacy tutor method preserved for existing tests."""
        messages = state.get("messages", [])
        last_query = messages[-1].get("content", "") if messages else ""
        task = Task(
            task_id="T1",
            worker="tutor",
            description=last_query,
            expected_output="Explanation",
            input_mapping={"query": last_query},
        )
        context = {
            "query": last_query,
            "profile": state.get("profile"),
            "semantic_memories": state.get("long_term_memories", []),
        }
        res = await self.workers["tutor"].execute(task, context)
        text = res.output.get("text", "") if res.output else ""
        return {
            "answer": text,
            "selected_provider": res.output.get("provider", "groq") if res.output else "groq",
            "selected_model": res.output.get("model", "default") if res.output else "default",
            "citations": res.citations,
        }

    async def document_retrieval(self, state: AgentState) -> Dict[str, Any]:
        """Legacy document retrieval preserved for existing tests."""
        query = state.get("search_query") or (state.get("messages", [{}])[-1].get("content", "") if state.get("messages") else "")
        task = Task(
            task_id="T1",
            worker="rag",
            description=query,
            expected_output="Chunks",
            input_mapping={"query": query, "document_id": state.get("document_id")},
        )
        context = {
            "query": query,
            "user_id": state.get("user_id", ""),
            "document_id": state.get("document_id"),
        }
        res = await self.workers["rag"].execute(task, context)
        chunks = res.output.get("chunks", []) if res.output else []
        return {
            "retrieved_chunks": chunks,
            "evidence_sufficient": len(chunks) > 0,
        }

    async def web_research(self, state: AgentState) -> Dict[str, Any]:
        """Legacy web research preserved for existing tests."""
        query = state.get("search_query") or (state.get("messages", [{}])[-1].get("content", "") if state.get("messages") else "")
        task = Task(
            task_id="T1",
            worker="research",
            description=query,
            expected_output="Articles",
            input_mapping={"query": query},
        )
        res = await self.workers["research"].execute(task, {"query": query})
        articles = res.output.get("articles", []) if res.output else []
        return {
            "research_sources": articles,
            "evidence_sufficient": len(articles) > 0,
        }

    async def hybrid_retrieval(self, state: AgentState) -> Dict[str, Any]:
        doc_res = await self.document_retrieval(state)
        web_res = await self.web_research(state)
        return {
            "retrieved_chunks": doc_res.get("retrieved_chunks", []),
            "research_sources": web_res.get("research_sources", []),
            "evidence_sufficient": len(doc_res.get("retrieved_chunks", [])) > 0 or len(web_res.get("research_sources", [])) > 0,
        }

    async def study_tools(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1].get("content", "") if messages else ""
        is_quiz = any(w in last_msg.lower() for w in ["quiz", "mcq", "test"])
        worker_key = "mcq" if is_quiz else "flashcard"
        task = Task(
            task_id="T1",
            worker=worker_key,
            description=last_msg,
            expected_output="Study items",
            input_mapping={"query": last_msg},
        )
        res = await self.workers[worker_key].execute(task, {"query": last_msg, "user_id": state.get("user_id", "")})
        text = res.output.get("text", "") if res.output else ""
        return {
            "answer": text,
            "citations": {},
            "selected_provider": "groq",
            "selected_model": "default",
        }

    async def synthesize(self, state: AgentState) -> Dict[str, Any]:
        chunks = state.get("retrieved_chunks", [])
        sources = state.get("research_sources", [])
        messages = state.get("messages", [])
        query = messages[-1].get("content", "") if messages else ""
        
        # Build synthesis context
        evidence_lines = []
        citations_meta = {"documents": [], "research_papers": []}
        if chunks:
            evidence_lines.append("=== UPLOADED TEXTBOOK EVIDENCE ===")
            for c in chunks:
                evidence_lines.append(f"Source [{c.get('filename')}, Page {c.get('page_number')}]:\n{c.get('content')}\n")
                citations_meta["documents"].append({
                    "filename": c.get("filename"),
                    "page_number": c.get("page_number"),
                    "chunk_id": c.get("chunk_id"),
                    "similarity_score": c.get("similarity_score"),
                })
        if sources:
            evidence_lines.append("=== SCHOLARLY RESEARCH ===")
            for s in sources:
                evidence_lines.append(f"Paper: {s.get('title')}\nAbstract: {s.get('abstract') or ''}\n")
                citations_meta["research_papers"].append({
                    "title": s.get("title"),
                    "authors": s.get("authors"),
                    "journal": s.get("journal"),
                    "pub_date": s.get("publication_date"),
                    "pmid": s.get("pmid"),
                    "doi": s.get("doi"),
                    "url": s.get("url"),
                    "evidence_level": s.get("evidence_level"),
                })

        evidence_text = "\n".join(evidence_lines)
        llm_messages = [
            LLMMessage(role="system", content=RESEARCH_SYNTHESIS_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"Student Question: {query}\n\nEVIDENCE:\n{evidence_text}"),
        ]
        res = await self.router.execute_with_fallback(
            task_type=TaskType.RESEARCH_SYNTHESIS,
            messages=llm_messages,
            complexity=ComplexityLevel.HIGH,
            temperature=0.4,
        )
        return {
            "answer": res.content,
            "citations": citations_meta,
            "selected_provider": res.provider.value,
            "selected_model": res.model,
        }

    async def verify_answer(self, state: AgentState) -> Dict[str, Any]:
        answer = state.get("answer", "")
        intent = state.get("intent", "tutor")
        chunks = state.get("retrieved_chunks", [])
        sources = state.get("research_sources", [])
        if intent in ("document_retrieval", "web_research", "hybrid_retrieval"):
            if not chunks and not sources:
                answer = "**Evidence Limitation Notice:**\nNo relevant passages found.\n\n" + answer
        return {"answer": answer}


StudyGraphNodes = DentalGraphNodes
