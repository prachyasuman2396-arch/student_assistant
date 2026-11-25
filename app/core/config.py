import os
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "AI Student Study & Research Assistant"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except Exception:
                return [v]
        return v

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/dental_agent_db"
    DATABASE_SYNC_URL: str = "postgresql+psycopg://postgres:postgrespassword@localhost:5432/dental_agent_db"

    # Security
    JWT_SECRET_KEY: str = "default-insecure-secret-key-change-in-production-12345"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # LLM Providers
    DEFAULT_LLM_PROVIDER: str = "groq"
    ENABLE_GEMINI: bool = True

    # Groq Settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_FAST_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_MAX_RETRIES: int = 3
    GROQ_TIMEOUT_SECONDS: float = 30.0

    # Gemini Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GEMINI_FLASH_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_TIMEOUT_SECONDS: float = 60.0

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "fastembed"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384

    # Multi-Extractor & Hybrid RAG Settings
    PYMUPDF_DPI: int = 200
    TRIAGE_SCANNED_WORD_FLOOR: int = 50
    TRIAGE_SCANNED_WORD_RATIO: float = 0.15
    TRIAGE_SCANNED_IMAGE_COVERAGE_PCT: float = 60.0
    EXTRACTION_CONCURRENCY: int = 3
    EXTRACTION_THREAD_POOL_SIZE: int = 4
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_DIMENSION: int = 1024
    EMBED_BATCH_SIZE: int = 16
    EMBED_USE_ONNX: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RETRIEVAL_DENSE_TOPK: int = 50
    RETRIEVAL_SPARSE_TOPK: int = 50
    RRF_K: int = 60
    RRF_FUSED_TOPK: int = 20
    FINAL_TOPK: int = 5
    GEMINI_VISION_MODEL: str = "gemini-1.5-pro"
    GEMINI_VISION_MAX_RETRIES: int = 2
    TABLE_EXTRACTOR: str = "docling"  # docling | marker | unstructured

    # Async Ingestion & Object Storage (Celery / Redis / S3)
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    S3_ENDPOINT_URL: str = ""
    S3_BUCKET: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    USE_ASYNC_INGESTION: bool = True

    # LangSmith Observability
    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "student_assistant"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # Document RAG Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    MAX_UPLOAD_SIZE_MB: int = 25
    UPLOAD_STORAGE_DIR: str = "./uploads"

    # Scholarly & Web Research
    NCBI_API_KEY: str = ""
    EUROPE_PMC_BASE_URL: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    CROSSREF_BASE_URL: str = "https://api.crossref.org"
    TAVILY_API_KEY: str = ""
    RESEARCH_MAX_RESULTS: int = 5


settings = Settings()
