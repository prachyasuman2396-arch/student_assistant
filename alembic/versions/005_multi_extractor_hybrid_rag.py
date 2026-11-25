"""Migration: multi-extractor ingestion, bge-m3 dense+sparse vectors, content_tsv trigger, HNSW/GIN indexes

Revision ID: 005_multi_extractor_hybrid_rag
Revises: 004_episodic_memories
Create Date: 2026-11-26 01:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector
from app.core.config import settings

# revision identifiers, used by Alembic.
revision = '005_multi_extractor_hybrid_rag'
down_revision = '004_episodic_memories'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Extend documents table with triage & async job status metadata
    op.add_column('documents', sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('documents', sa.Column('last_error', sa.JSON(), nullable=True))
    op.add_column('documents', sa.Column('triage_summary', sa.JSON(), nullable=True))

    # 2. Extend document_chunks table for multi-extractor & bge-m3 hybrid RAG
    op.add_column('document_chunks', sa.Column('bbox', sa.JSON(), nullable=True))
    op.add_column('document_chunks', sa.Column('chunk_type', sa.String(length=32), nullable=False, server_default='text'))
    op.add_column('document_chunks', sa.Column('extractor', sa.String(length=16), nullable=False, server_default='pymupdf'))
    op.add_column('document_chunks', sa.Column('content_tsv', postgresql.TSVECTOR(), nullable=True))
    op.add_column('document_chunks', sa.Column('dense_embedding', Vector(settings.EMBED_DIMENSION), nullable=True))
    op.add_column('document_chunks', sa.Column('sparse_values', sa.JSON(), nullable=True))
    op.add_column('document_chunks', sa.Column('heading_path', sa.Text(), nullable=True))
    op.add_column('document_chunks', sa.Column('extraction_meta', sa.JSON(), nullable=True))

    # 3. B-tree index on (user_id, document_id, page_number)
    op.create_index(
        'ix_doc_chunk_user_doc_page',
        'document_chunks',
        ['user_id', 'document_id', 'page_number'],
        unique=False,
    )

    # 4. GIN index on content_tsv for BM25 / ts_rank_cd sparse retrieval
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_chunks_content_tsv "
        "ON document_chunks USING gin (content_tsv);"
    )

    # 5. HNSW index on dense_embedding (m=16, ef_construction=200)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_chunks_dense_hnsw "
        "ON document_chunks USING hnsw (dense_embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200);"
    )

    # 6. Trigger to populate content_tsv automatically on INSERT/UPDATE
    op.execute("""
        CREATE OR REPLACE FUNCTION document_chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        DROP TRIGGER IF EXISTS trg_document_chunks_tsv ON document_chunks;
        CREATE TRIGGER trg_document_chunks_tsv
        BEFORE INSERT OR UPDATE OF content ON document_chunks
        FOR EACH ROW EXECUTE FUNCTION document_chunks_tsv_trigger();
    """)

    # Backfill existing rows
    op.execute("UPDATE document_chunks SET content_tsv = to_tsvector('english', COALESCE(content, '')) WHERE content_tsv IS NULL;")

    # 7. Postgres function for sparse lexical dot-product scoring over JSONB
    op.execute("""
        CREATE OR REPLACE FUNCTION sparse_score(query_sparse JSONB, chunk_sparse JSONB)
        RETURNS DOUBLE PRECISION AS $$
        DECLARE
            score DOUBLE PRECISION := 0.0;
            k TEXT;
            q_val DOUBLE PRECISION;
            c_val DOUBLE PRECISION;
        BEGIN
            IF query_sparse IS NULL OR chunk_sparse IS NULL THEN
                RETURN 0.0;
            END IF;
            FOR k, q_val IN SELECT key, value::DOUBLE PRECISION FROM jsonb_each_text(query_sparse)
            LOOP
                IF chunk_sparse ? k THEN
                    c_val := (chunk_sparse ->> k)::DOUBLE PRECISION;
                    score := score + (q_val * c_val);
                END IF;
            END LOOP;
            RETURN score;
        END
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS sparse_score(JSONB, JSONB);")
    op.execute("DROP TRIGGER IF EXISTS trg_document_chunks_tsv ON document_chunks;")
    op.execute("DROP FUNCTION IF EXISTS document_chunks_tsv_trigger();")
    op.execute("DROP INDEX IF EXISTS ix_doc_chunks_dense_hnsw;")
    op.execute("DROP INDEX IF EXISTS ix_doc_chunks_content_tsv;")
    op.drop_index('ix_doc_chunk_user_doc_page', table_name='document_chunks')

    op.drop_column('document_chunks', 'extraction_meta')
    op.drop_column('document_chunks', 'heading_path')
    op.drop_column('document_chunks', 'sparse_values')
    op.drop_column('document_chunks', 'dense_embedding')
    op.drop_column('document_chunks', 'content_tsv')
    op.drop_column('document_chunks', 'extractor')
    op.drop_column('document_chunks', 'chunk_type')
    op.drop_column('document_chunks', 'bbox')

    op.drop_column('documents', 'triage_summary')
    op.drop_column('documents', 'last_error')
    op.drop_column('documents', 'chunk_count')
