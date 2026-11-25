from unittest.mock import AsyncMock, MagicMock
import pytest

from app.rag.retriever import (
    HybridRAGRetriever,
    HybridRetrievedChunk,
    compute_rrf_scores,
    compute_sparse_dot_product,
)


def _make_chunk(cid: str, content: str, page: int = 1, doc_id: str = "doc-1", score: float = 0.5) -> HybridRetrievedChunk:
    return HybridRetrievedChunk(
        chunk_id=cid,
        document_id=doc_id,
        filename="Endodontics_Text.pdf",
        page_number=page,
        content=content,
        similarity_score=score,
        chunk_type="text",
        extractor="pymupdf",
        heading_path="Ch.7 > Pulpitis",
        citation_label=f"[Endodontics_Text.pdf, p.{page}, text]",
    )


def test_rrf_ordering_arithmetic():
    dense = ["c1", "c2", "c3"]
    sparse = ["c3", "c1", "c4"]
    fused = compute_rrf_scores([dense, sparse], k=60)
    score_map = dict(fused)

    # c1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    assert score_map["c1"] == pytest.approx((1.0 / 61.0) + (1.0 / 62.0), rel=1e-6)
    # c3: 1/(60+3) + 1/(60+1) = 1/63 + 1/61
    assert score_map["c3"] == pytest.approx((1.0 / 63.0) + (1.0 / 61.0), rel=1e-6)
    assert fused[0][0] == "c1"
    assert fused[1][0] == "c3"


def test_sparse_dot_product_arithmetic():
    q_sparse = {10: 0.8, 25: 0.5, 99: 0.2}
    c_sparse = {"10": 0.75, "25": 0.4, "77": 0.9}
    score = compute_sparse_dot_product(q_sparse, c_sparse)
    # 0.8*0.75 + 0.5*0.4 = 0.6 + 0.2 = 0.8
    assert score == pytest.approx(0.8, rel=1e-6)


@pytest.mark.asyncio
async def test_hybrid_retrieval_rrf_rerank_and_top5():
    mock_session = AsyncMock()
    retriever = HybridRAGRetriever(session=mock_session)

    # Seed 8 dense and 8 sparse candidates
    dense_chunks = [_make_chunk(f"c{i}", f"Generic dental passage {i}", page=i) for i in range(1, 9)]
    # Make c7 highly relevant to the exact query so cross-encoder reranker promotes it
    dense_chunks[6].content = "Irreversible pulpitis diagnostic criteria: lingering thermal pain and spontaneous pain."
    sparse_chunks = list(reversed(dense_chunks))

    retriever._encode_query = MagicMock(return_value=([0.1] * 1024, {1: 0.9}))
    retriever._dense_retrieve = AsyncMock(return_value=dense_chunks)
    retriever._sparse_retrieve = AsyncMock(return_value=sparse_chunks)

    results = await retriever.retrieve(
        query="irreversible pulpitis diagnostic criteria lingering thermal pain",
        user_id="user-tenant-A",
        top_k=5,
    )

    assert len(results) == 5
    # Cross-encoder rerank promotes c7 to top position deterministically
    assert results[0].chunk_id == "c7"
    assert results[0].citation_label == "[Endodontics_Text.pdf, p.7, text]"


@pytest.mark.asyncio
async def test_cross_tenant_retrieval_returns_zero_rows():
    mock_session = AsyncMock()
    # Empty result when querying for user-tenant-B
    mock_exec_res = MagicMock()
    mock_exec_res.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_exec_res)

    retriever = HybridRAGRetriever(session=mock_session)
    retriever._encode_query = MagicMock(return_value=([0.1] * 1024, {1: 0.5}))

    results = await retriever.retrieve(
        query="irreversible pulpitis",
        user_id="user-tenant-B",
    )

    assert results == []
    # Verify execute was called twice (dense + sparse) and both queries carried user_id filtering
    assert mock_session.execute.call_count == 2
    for call in mock_session.execute.call_args_list:
        compiled = str(call.args[0])
        assert "document_chunks.user_id" in compiled
