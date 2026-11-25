from celery import Celery
from kombu import Exchange, Queue
from app.core.config import settings

celery_app = Celery(
    "student_assistant_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

default_exchange = Exchange("ingestion", type="direct")
dlq_exchange = Exchange("ingestion_dlq", type="direct")

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_queues=(
        Queue("ingestion", exchange=default_exchange, routing_key="ingestion"),
        Queue("dlq", exchange=dlq_exchange, routing_key="dlq"),
    ),
    task_default_queue="ingestion",
    task_default_exchange="ingestion",
    task_default_routing_key="ingestion",
    task_routes={
        "app.workers.tasks.ingest_document_task": {"queue": "ingestion"},
        "app.workers.tasks.dead_letter_handler": {"queue": "dlq"},
    },
)
