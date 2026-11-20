import asyncio
import time
from typing import Dict, Any, List, Set, Optional
from app.graph.state import Task, TaskResult, TaskStatus, ExecutionPlan
from app.graph.workers.base import BaseWorker
from app.core.logging import logger


class Orchestrator:
    """
    Deterministic dependency-aware task orchestrator.
    Schedules and dispatches independent tasks concurrently, tracks task lifecycle states,
    propagates upstream outputs to dependent workers, detects blocked tasks, and handles retries.
    This component is strictly algorithmic Python logic and contains NO LLM calls.
    """

    def __init__(
        self,
        workers: Dict[str, BaseWorker],
        max_concurrency: int = 5,
        default_max_retries: int = 2,
        task_timeout_seconds: float = 60.0,
    ):
        self.workers = workers
        self.max_concurrency = max_concurrency
        self.default_max_retries = default_max_retries
        self.task_timeout_seconds = task_timeout_seconds

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any],
    ) -> List[TaskResult]:
        """
        Executes an ExecutionPlan DAG to completion or block state.
        Dispatches all independent tasks in parallel waves.
        """
        tasks_map: Dict[str, Task] = {t.task_id: t for t in plan.tasks}
        task_results: Dict[str, TaskResult] = {}
        completed_ids: Set[str] = set()
        failed_ids: Set[str] = set()
        blocked_ids: Set[str] = set()
        dispatched_ids: Set[str] = set()

        semaphore = asyncio.Semaphore(self.max_concurrency)

        while True:
            # 1. Identify blocked tasks (any dependency failed or blocked)
            newly_blocked = False
            for tid, task in tasks_map.items():
                if task.status == TaskStatus.PENDING:
                    if any(dep in failed_ids or dep in blocked_ids for dep in task.dependencies):
                        task.status = TaskStatus.BLOCKED
                        blocked_ids.add(tid)
                        newly_blocked = True
                        task_results[tid] = TaskResult(
                            task_id=tid,
                            worker=task.worker,
                            status=TaskStatus.BLOCKED,
                            output=None,
                            error=f"Blocked: Upstream dependency failed in {task.dependencies}.",
                        )
                        logger.warning(f"Task '{tid}' marked as BLOCKED due to upstream failure.")

            # 2. Identify executable tasks
            # Status is PENDING and all dependencies are in completed_ids
            executable_tasks: List[Task] = []
            for tid, task in tasks_map.items():
                if task.status == TaskStatus.PENDING and tid not in dispatched_ids:
                    if all(dep in completed_ids for dep in task.dependencies):
                        executable_tasks.append(task)

            # If no tasks are executable, check termination
            if not executable_tasks:
                all_done = all(
                    t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED)
                    for t in tasks_map.values()
                )
                if all_done:
                    break
                # If there are still pending tasks but none can be executed and no active workers, break deadlock
                if not newly_blocked:
                    for tid, task in tasks_map.items():
                        if task.status == TaskStatus.PENDING:
                            task.status = TaskStatus.BLOCKED
                            blocked_ids.add(tid)
                            task_results[tid] = TaskResult(
                                task_id=tid,
                                worker=task.worker,
                                status=TaskStatus.BLOCKED,
                                output=None,
                                error="Deadlock detected: unresolved dependencies.",
                            )
                    break

            # 3. Dispatch ready tasks concurrently
            async def run_single_task(task_to_run: Task) -> TaskResult:
                async with semaphore:
                    task_to_run.status = TaskStatus.RUNNING
                    dispatched_ids.add(task_to_run.task_id)

                    worker = self.workers.get(task_to_run.worker)
                    if not worker:
                        task_to_run.status = TaskStatus.FAILED
                        return TaskResult(
                            task_id=task_to_run.task_id,
                            worker=task_to_run.worker,
                            status=TaskStatus.FAILED,
                            error=f"Unregistered worker '{task_to_run.worker}'.",
                        )

                    # Pass upstream task results
                    task_ctx = {
                        **context,
                        "upstream_results": {
                            dep_id: task_results[dep_id]
                            for dep_id in task_to_run.dependencies
                            if dep_id in task_results
                        },
                    }

                    # Execute with bounded retries and timeout
                    attempts = 0
                    max_attempts = getattr(task_to_run, "max_retries", self.default_max_retries) + 1

                    while attempts < max_attempts:
                        attempts += 1
                        try:
                            logger.info(
                                f"[Orchestrator] Executing task '{task_to_run.task_id}' "
                                f"with worker '{task_to_run.worker}' (attempt {attempts}/{max_attempts})"
                            )
                            result = await asyncio.wait_for(
                                worker.execute(task_to_run, task_ctx),
                                timeout=self.task_timeout_seconds,
                            )
                            if result.status == TaskStatus.COMPLETED:
                                task_to_run.status = TaskStatus.COMPLETED
                                return result
                            else:
                                logger.warning(
                                    f"[Orchestrator] Task '{task_to_run.task_id}' returned non-completed status: "
                                    f"{result.status} (error: {result.error})"
                                )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"[Orchestrator] Task '{task_to_run.task_id}' timed out after {self.task_timeout_seconds}s"
                            )
                            result = TaskResult(
                                task_id=task_to_run.task_id,
                                worker=task_to_run.worker,
                                status=TaskStatus.FAILED,
                                error=f"Task timed out after {self.task_timeout_seconds}s",
                            )
                        except Exception as e:
                            logger.error(
                                f"[Orchestrator] Task '{task_to_run.task_id}' error: {e}",
                                exc_info=True,
                            )
                            result = TaskResult(
                                task_id=task_to_run.task_id,
                                worker=task_to_run.worker,
                                status=TaskStatus.FAILED,
                                error=str(e),
                            )

                    # Exhausted attempts
                    task_to_run.status = TaskStatus.FAILED
                    return result

            # Run batch of ready tasks concurrently
            wave_results = await asyncio.gather(
                *(run_single_task(t) for t in executable_tasks),
                return_exceptions=False,
            )

            # Update tracking states
            for res in wave_results:
                task_results[res.task_id] = res
                if res.status == TaskStatus.COMPLETED:
                    completed_ids.add(res.task_id)
                else:
                    failed_ids.add(res.task_id)

        # Return results ordered by task_id appearance in plan
        ordered_results = [
            task_results[t.task_id]
            for t in plan.tasks
            if t.task_id in task_results
        ]
        return ordered_results
