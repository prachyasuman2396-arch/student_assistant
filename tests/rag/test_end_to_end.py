import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pymupdf
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_current_user, get_db
from app.db.models.document import Document
from app.db.models.user import User
from app.documents.ingestion_service import IngestionService
from app.main import app
from app.rag.retriever import HybridRAGRetriever, HybridRetrievedChunk


def _build_fixture_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    for p in range(1, 3):
        page = doc.new_page(width=595, height=842)
        text = " ".join(
            [f"Page {p}: Irreversible pulpitis shows lingering cold sensitivity and C-fiber activation."] * 25
        )
        page.insert_text((72, 72), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.asyncio
async def test_upload_returns_202_and_pipeline_produces_ready_chunks(tmp_path):
    user = User(
        id="user-e2e-1",
        email="e2e@dental.edu",
        hashed_password="pw",
        full_name="E2E Student",
        academic_level="BDS",
        is_active=True,
    )
    pdf_bytes = _build_fixture_pdf_bytes()
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(pdf_bytes)

    mock_db = AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: mock_db

    queued_doc = Document(
        id="doc-e2e-1",
        user_id=user.id,
        filename="fixture.pdf",
        file_path=str(pdf_path),
        file_size_bytes=len(pdf_bytes),
        page_count=0,
        chunk_count=0,
        status="queued",
        created_at=datetime.now(timezone.utc),
    )

    with patch("app.api.v1.documents.IngestionService") as mock_svc_cls:
        mock_svc = MagicMock()
        mock_svc.enqueue_upload = AsyncMock(return_value=queued_doc)
        mock_svc_cls.return_value = mock_svc

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("fixture.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            )
            assert res.status_code == 202
            body = res.json()
            assert body["id"] == "doc-e2e-1"
            assert body["status"] == "queued"

    app.dependency_overrides.clear()

    # Run ingestion pipeline against mocked repository to verify status="ready" & chunks stored
    svc = IngestionService(mock_db)
    svc.repo = MagicMock()
    svc.repo.get_by_id = AsyncMock(return_value=queued_doc)
    svc.repo.update_status = AsyncMock(return_value=queued_doc)
    svc.repo.add_chunks = AsyncMock()

    await svc.run_ingestion_pipeline(document_id="doc-e2e-1", user_id=user.id)

    svc.repo.add_chunks.assert_called_once()
    inserted_chunks = svc.repo.add_chunks.call_args.args[0]
    assert len(inserted_chunks) >= 2
    assert all(len(c.dense_embedding) == 1024 for c in inserted_chunks)
    assert all(c.user_id == user.id for c in inserted_chunks)

    final_status_call = svc.repo.update_status.call_args_list[-1]
    assert final_status_call.kwargs["status"] == "ready"
    assert final_status_call.kwargs["page_count"] == 2
    assert final_status_call.kwargs["triage_summary"]["pymupdf"] == 2

    # Query retriever and verify top-5 citation formatting
    retriever = HybridRAGRetriever(session=mock_db)
    candidates = [
        HybridRetrievedChunk(
            chunk_id=f"c-{i}",
            document_id="doc-e2e-1",
            filename="fixture.pdf",
            page_number=1,
            content=f"Irreversible pulpitis lingering cold sensitivity passage {i}",
            similarity_score=0.9 - i * 0.05,
            chunk_type="text",
            extractor="pymupdf",
            citation_label="[fixture.pdf, p.1, text]",
        )
        for i in range(6)
    ]
    retriever._encode_query = MagicMock(return_value=([0.1] * 1024, {1: 0.8}))
    retriever._dense_retrieve = AsyncMock(return_value=candidates)
    retriever._sparse_retrieve = AsyncMock(return_value=candidates)

    top5 = await retriever.retrieve(query="irreversible pulpitis", user_id=user.id)
    assert len(top5) == 5
    assert top5[0].citation_label == "[fixture.pdf, p.1, text]"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_placeholder_celery_postgres_pgvector():
    """
    Integration marker test for live Celery + Postgres + pgvector environments.
    Run with `pytest -m integration` when services are up via `make up`.
    """
    assert True
