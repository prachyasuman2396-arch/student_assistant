import pytest
from app.graph.validator import PlanValidator
from app.graph.state import ExecutionPlan, Task, TaskStatus


def test_validator_valid_dag_plan():
    validator = PlanValidator()
    plan = ExecutionPlan(
        objective="Valid DAG Plan",
        tasks=[
            Task(task_id="T1", worker="rag", description="Fetch doc", dependencies=[], expected_output="Chunks"),
            Task(task_id="T2", worker="research", description="Fetch papers", dependencies=[], expected_output="Papers"),
            Task(task_id="T3", worker="tutor", description="Synthesize", dependencies=["T1", "T2"], expected_output="Synthesis"),
            Task(task_id="T4", worker="mcq", description="Generate MCQs", dependencies=["T3"], expected_output="MCQs"),
        ],
    )
    is_valid, error = validator.validate(plan)
    assert is_valid is True
    assert error is None


def test_validator_detects_duplicate_task_ids():
    validator = PlanValidator()
    plan = ExecutionPlan(
        objective="Duplicate IDs",
        tasks=[
            Task(task_id="T1", worker="rag", description="Fetch doc", dependencies=[], expected_output="Chunks"),
            Task(task_id="T1", worker="research", description="Fetch papers", dependencies=[], expected_output="Papers"),
        ],
    )
    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "Duplicate task_id detected" in error


def test_validator_detects_unregistered_worker():
    validator = PlanValidator()
    plan = ExecutionPlan(
        objective="Invalid worker",
        tasks=[
            Task(task_id="T1", worker="unregistered_crypto_miner", description="Mine", dependencies=[], expected_output="Coins"),
        ],
    )
    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "unregistered worker" in error


def test_validator_detects_self_dependency():
    validator = PlanValidator()
    plan = ExecutionPlan(
        objective="Self dependency",
        tasks=[
            Task(task_id="T1", worker="tutor", description="Self loop", dependencies=["T1"], expected_output="Result"),
        ],
    )
    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "Self-dependency detected" in error


def test_validator_detects_nonexistent_dependency():
    validator = PlanValidator()
    plan = ExecutionPlan(
        objective="Missing dependency",
        tasks=[
            Task(task_id="T1", worker="tutor", description="Task 1", dependencies=["T99"], expected_output="Result"),
        ],
    )
    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "non-existent dependency" in error


def test_validator_detects_cyclic_dependency():
    validator = PlanValidator()
    plan = ExecutionPlan(
        objective="Cycle A -> B -> C -> A",
        tasks=[
            Task(task_id="T1", worker="rag", description="Task 1", dependencies=["T3"], expected_output="A"),
            Task(task_id="T2", worker="research", description="Task 2", dependencies=["T1"], expected_output="B"),
            Task(task_id="T3", worker="tutor", description="Task 3", dependencies=["T2"], expected_output="C"),
        ],
    )
    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "Cyclic dependency detected" in error


def test_validator_exceeding_max_tasks_limit():
    validator = PlanValidator(max_tasks=3)
    plan = ExecutionPlan(
        objective="Too many tasks",
        tasks=[
            Task(task_id=f"T{i}", worker="tutor", description=f"Task {i}", dependencies=[], expected_output="X")
            for i in range(1, 5)
        ],
    )
    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "exceeds maximum limit" in error
