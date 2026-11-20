import pytest
import asyncio
from typing import Dict, Any
from app.graph.orchestrator import Orchestrator
from app.graph.workers.base import BaseWorker
from app.graph.state import ExecutionPlan, Task, TaskResult, TaskStatus


class MockEchoWorker(BaseWorker):
    def __init__(self, worker_name: str, delay: float = 0.01, should_fail: bool = False):
        self._name = worker_name
        self.delay = delay
        self.should_fail = should_fail
        self.execution_order = []
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        self.call_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.FAILED,
                error="Simulated worker failure",
            )

        # Inspect upstream results
        upstream = context.get("upstream_results", {})
        upstream_ids = list(upstream.keys())

        return TaskResult(
            task_id=task.task_id,
            worker=self.name,
            status=TaskStatus.COMPLETED,
            output={"text": f"Output from {self.name} for {task.task_id}", "upstream": upstream_ids},
        )


@pytest.mark.asyncio
async def test_orchestrator_parallel_execution():
    w1 = MockEchoWorker("rag", delay=0.05)
    w2 = MockEchoWorker("research", delay=0.05)
    workers = {"rag": w1, "research": w2}

    orchestrator = Orchestrator(workers=workers, max_concurrency=5)
    plan = ExecutionPlan(
        objective="Parallel RAG & Research",
        tasks=[
            Task(task_id="T1", worker="rag", description="Fetch doc", dependencies=[], expected_output="Chunks"),
            Task(task_id="T2", worker="research", description="Fetch papers", dependencies=[], expected_output="Papers"),
        ],
    )

    start_time = asyncio.get_event_loop().time()
    results = await orchestrator.execute_plan(plan, {})
    duration = asyncio.get_event_loop().time() - start_time

    assert len(results) == 2
    assert all(r.status == TaskStatus.COMPLETED for r in results)
    # If executed concurrently, total time should be close to ~0.05s, definitely under 0.09s
    assert duration < 0.09


@pytest.mark.asyncio
async def test_orchestrator_sequential_and_dependent_execution():
    w1 = MockEchoWorker("rag", delay=0.01)
    w2 = MockEchoWorker("research", delay=0.01)
    w3 = MockEchoWorker("tutor", delay=0.01)
    workers = {"rag": w1, "research": w2, "tutor": w3}

    orchestrator = Orchestrator(workers=workers)
    plan = ExecutionPlan(
        objective="Dependent execution",
        tasks=[
            Task(task_id="T1", worker="rag", description="RAG", dependencies=[], expected_output="A"),
            Task(task_id="T2", worker="research", description="Research", dependencies=[], expected_output="B"),
            Task(task_id="T3", worker="tutor", description="Synthesize", dependencies=["T1", "T2"], expected_output="C"),
        ],
    )

    results = await orchestrator.execute_plan(plan, {})
    assert len(results) == 3
    assert all(r.status == TaskStatus.COMPLETED for r in results)

    # Verify T3 received outputs from T1 and T2
    t3_res = [r for r in results if r.task_id == "T3"][0]
    assert set(t3_res.output["upstream"]) == {"T1", "T2"}


@pytest.mark.asyncio
async def test_orchestrator_handles_worker_failure_and_blocks_dependents():
    w1 = MockEchoWorker("rag", should_fail=True)
    w2 = MockEchoWorker("research")
    w3 = MockEchoWorker("tutor")
    workers = {"rag": w1, "research": w2, "tutor": w3}

    orchestrator = Orchestrator(workers=workers, default_max_retries=1)
    plan = ExecutionPlan(
        objective="Failure propagation",
        tasks=[
            Task(task_id="T1", worker="rag", description="Will fail", dependencies=[], expected_output="A"),
            Task(task_id="T2", worker="research", description="Will pass", dependencies=[], expected_output="B"),
            Task(task_id="T3", worker="tutor", description="Should be blocked", dependencies=["T1"], expected_output="C"),
        ],
    )

    results = await orchestrator.execute_plan(plan, {})
    results_by_id = {r.task_id: r for r in results}

    assert results_by_id["T1"].status == TaskStatus.FAILED
    assert results_by_id["T2"].status == TaskStatus.COMPLETED
    assert results_by_id["T3"].status == TaskStatus.BLOCKED
    assert "Blocked" in results_by_id["T3"].error
    # T3 worker should never have been executed
    assert w3.call_count == 0


@pytest.mark.asyncio
async def test_orchestrator_prevents_duplicate_dispatch():
    w1 = MockEchoWorker("tutor")
    workers = {"tutor": w1}

    orchestrator = Orchestrator(workers=workers)
    plan = ExecutionPlan(
        objective="No duplicate execution",
        tasks=[
            Task(task_id="T1", worker="tutor", description="Task 1", dependencies=[], expected_output="A"),
        ],
    )

    results = await orchestrator.execute_plan(plan, {})
    assert len(results) == 1
    assert w1.call_count == 1
