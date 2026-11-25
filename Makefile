.PHONY: up down migrate test test-integration eval ingest

up:
	docker-compose up --build -d

down:
	docker-compose down

migrate:
	alembic upgrade head

test:
	.venv/bin/pytest -v -m "not slow and not integration"

test-integration:
	.venv/bin/pytest -v -m "integration"

eval:
	.venv/bin/python -m evaluation.run_eval --dataset evaluation/dental_eval_dataset.json

ingest:
	@if [ -z "$(FILE)" ]; then echo "Usage: make ingest FILE=path/to/doc.pdf"; exit 1; fi
	.venv/bin/python -c "import asyncio, sys; from app.documents.triage import PageTriager; res = PageTriager.triage_pdf('$(FILE)'); print(res.model_dump_json(indent=2))"
