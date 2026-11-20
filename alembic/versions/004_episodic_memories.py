"""Migration: episodic_memories table with vector support

Revision ID: 004_episodic_memories
Revises: 003_rag_pipeline_metadata
Create Date: 2026-11-20 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from app.core.config import settings

# revision identifiers, used by Alembic.
revision = '004_episodic_memories'
down_revision = '003_rag_pipeline_metadata'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'episodic_memories',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=True),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('topic', sa.String(length=255), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('details_json', sa.JSON(), nullable=True),
        sa.Column('importance', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('embedding', Vector(settings.EMBEDDING_DIMENSION), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_episodic_memories_user_id', 'episodic_memories', ['user_id'], unique=False)
    op.create_index('ix_episodic_memories_session_id', 'episodic_memories', ['session_id'], unique=False)
    op.create_index('ix_episodic_memories_event_type', 'episodic_memories', ['event_type'], unique=False)
    op.create_index('ix_episodic_memories_topic', 'episodic_memories', ['topic'], unique=False)
    op.create_index('ix_episodic_memories_user_type', 'episodic_memories', ['user_id', 'event_type'], unique=False)
    op.create_index('ix_episodic_memories_user_topic', 'episodic_memories', ['user_id', 'topic'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_episodic_memories_user_topic', table_name='episodic_memories')
    op.drop_index('ix_episodic_memories_user_type', table_name='episodic_memories')
    op.drop_index('ix_episodic_memories_topic', table_name='episodic_memories')
    op.drop_index('ix_episodic_memories_event_type', table_name='episodic_memories')
    op.drop_index('ix_episodic_memories_session_id', table_name='episodic_memories')
    op.drop_index('ix_episodic_memories_user_id', table_name='episodic_memories')
    op.drop_table('episodic_memories')
