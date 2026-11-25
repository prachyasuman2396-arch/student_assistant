# Dental AI Study & Research Assistant 🦷🤖

A production-oriented, ChatGPT-like pedagogical study and research assistant designed specifically for dental students (BDS, DMD/DDS, dental residents, and postgraduates).

The system features a **Planner–Orchestrator–Multi-Agent Architecture with Persistent Memory** using LangGraph, an intelligent dual-LLM router (Groq + Google Gemini), pgvector-backed semantic document RAG (using PyMuPDF), live scholarly biomedical research retrieval (NCBI PubMed & Europe PMC), active-recall study tools (MCQs and flashcards), and three-tier persistent memory (short-term thread checkpointing, semantic profile preferences, and episodic learning events).

---

## 🏛️ System Architecture

```
Student Query + Context
          │
          ▼
┌────────────────────────────────────────────────────────┐
│               1. Memory Retrieval Node                 │
│  - Short-Term: Thread checkpoints & message window     │
│  - Semantic: User preferences, learning goals (pgvector)│
│  - Episodic: Relevant past quiz scores & struggles     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                  2. Planner Agent                      │
│  - Dynamic task decomposition using Pydantic schema    │
│  - Determines workers, dependencies, parallelizability │
│  - Direct execution bypass for trivial single-turn Qs  │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                 3. Plan Validator                      │
│  - Validates DAG: cycle detection (Kahn's / DFS)       │
│  - Validates registered workers & dependency IDs       │
│  - Bounded retry loop (max 2 revisions) on invalid plan│
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│           4. Deterministic Orchestrator                │
│  - Evaluates dependency readiness in execution state   │
│  - Dispatches executable tasks concurrently in parallel │
│  - Enforces concurrency limits, timeouts & retries     │
└──────┬──────────────────┬───────────────────┬──────────┘
       │                  │                   │
       ▼ (Parallel)       ▼ (Parallel)        ▼ (Sequential / Dependent)
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  RAG Worker  │   │Research Worker│  │ Tutor Worker │ ... (MCQ / Flashcard)
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│              5. Result Aggregator Node                 │
│  - Collects worker outputs (with state reducer)        │
│  - Merges evidence dossier & citation metadata         │
│  - Synthesizes final pedagogical response              │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                6. Memory Update Node                   │
│  - Semantic Extractor: Discovers new goals/preferences │
│  - Episodic Logger: Records study events & struggles   │
│  - Strict tenant isolation & deduplication             │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
                     Final Answer
```

### Key Pillars
1. **Intelligent Dual-LLM Router:**
   - **Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`):** Ultra-fast definitions, query classification, lightweight MCQ/flashcard generation, and conversational follow-ups.
   - **Google Gemini (`gemini-1.5-pro`, `gemini-1.5-flash`):** High-context multi-paper research synthesis, complex biological reasoning, and multimodal dental imagery.
   - **Bounded Fallback & Retries:** Exponential backoff for rate limits (`429`) and transient errors, graceful cross-provider fallback.
2. **PostgreSQL & pgvector Vector Storage:**
   - Cosine distance similarity search (`<=>`) on uploaded PDF chunks.
   - Strict multi-tenant user isolation enforced on every query.
   - 1-indexed page-level citation tracking (`[Cohen's Pathways of the Pulp, p. 142]`).
3. **Scholarly Biomedical Literature:**
   - Real-time querying against NCBI PubMed E-utilities and Europe PMC REST APIs.
   - Priority ranking of Systematic Reviews, Meta-Analyses, and Clinical Practice Guidelines.
   - Sanitization of untrusted research text to prevent prompt injection.
4. **Active-Recall Study Engine:**
   - Board exam MCQs (INBDE, NDEB, BDS) with detailed rationales for correct answers and distractors.
   - High-yield active-recall flashcards persisted in PostgreSQL.
5. **Adaptive Multi-Extractor Ingestion & Hybrid RAG:**
   - **Page-Level Triage** (`PyMuPDF` first pass) routes pages to `pymupdf` (text), `docling` (tables, with Marker/Unstructured fallback), `gemini` (200 DPI vision for scans/diagrams), or `mixed`.
   - **Bi-Encoder + Hybrid Retrieval**: `BAAI/bge-m3` dense (1024-dim HNSW) + sparse BM25 (`content_tsv` GIN `ts_rank_cd`) fused with RRF (`k=60`) and reranked with `BAAI/bge-reranker-v2-m3` to top-5 chunks.
   - **Async Celery + Redis Queue**: `/api/v1/documents/upload` returns `202 Accepted` immediately with SHA-256 + `user_id` idempotency.
   - **LangSmith Observability**: Full span tracing from API entry through retrieval steps and LLM calls.

### Ingestion & Retrieval Pipelines
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full details.

```
PDF Upload (202 queued) ─► Page Triage (PyMuPDF) ─┬─► pymupdf (text)
                                                  ├─► docling (tables -> MD)
                                                  ├─► gemini (200 DPI vision)
                                                  └─► mixed (merged in order)
  ─► Adaptive Chunker ─► BAAI/bge-m3 (dense+sparse) ─► pgvector (HNSW + GIN)

Query ─► bge-m3 ─┬─► Dense Top-50 (pgvector <=>)
                 └─► Sparse Top-50 (ts_rank_cd)
                   ─► RRF (k=60, top-20) ─► bge-reranker-v2-m3 ─► Top-5 ─► LLM
```

### Makefile Targets & Evaluation
- `make up`: Start Postgres+pgvector, Redis, Celery worker, and FastAPI (`docker-compose up --build -d`).
- `make migrate`: Run `alembic upgrade head`.
- `make test`: Run fast unit tests (`pytest -v -m "not slow and not integration"`).
- `make eval`: Run the 30-item golden dental Q&A evaluation harness (`evaluation/run_eval.py`) measuring faithfulness, citation precision, and answer relevancy.
- `make ingest FILE=path/to/book.pdf`: Run page-level triage on a local PDF.

### Additional Environment Variables
See `.env.example` for all tunables (`PYMUPDF_DPI`, `EXTRACTION_CONCURRENCY`, `EMBED_MODEL`, `RERANK_MODEL`, `CELERY_BROKER_URL`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose (or a local PostgreSQL instance with `pgvector` enabled)

### 2. Environment Setup
Clone the repository and create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template:
```bash
cp .env.example .env
```
Fill in your credentials in `.env`:
- `GROQ_API_KEY`: Your Groq API key.
- `GEMINI_API_KEY`: Your Google Gemini API key.
- `DATABASE_URL`: PostgreSQL async connection string.
- `DATABASE_SYNC_URL`: PostgreSQL sync connection string (for Alembic).

---

## 🐳 Docker Deployment

The simplest way to run the entire stack (PostgreSQL with `pgvector` + FastAPI Backend) is with Docker Compose:

```bash
docker-compose up --build -d
```

Check logs:
```bash
docker-compose logs -f backend
```

Once running, access:
- **API Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🗄️ Database Setup & Migrations

If running locally without Docker:
1. Ensure PostgreSQL is running on port 5432 with `vector` extension available:
   ```sql
   CREATE DATABASE dental_agent_db;
   \c dental_agent_db;
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. Run Alembic migrations:
   ```bash
   source .venv/bin/activate
   alembic upgrade head
   ```

---

## 🧪 Running Tests

The test suite includes unit tests for LLM routing, fallback policies, prompt sanitization, PDF parsing, pgvector tenant isolation, and API endpoints:

```bash
source .venv/bin/activate
pytest -v
```

---

## 📚 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Application, database, pgvector, and LLM readiness checks |
| `POST` | `/api/v1/auth/register` | Register a new dental student account |
| `POST` | `/api/v1/auth/login` | Authenticate and obtain JWT access token |
| `GET` | `/api/v1/auth/me` | Fetch authenticated student profile |
| `POST` | `/api/v1/conversations` | Create a new study session thread |
| `GET` | `/api/v1/conversations` | List conversation threads for current student |
| `GET` | `/api/v1/conversations/{id}/messages` | Retrieve message history of a conversation |
| `POST` | `/api/v1/chat` | Send student query (supports text, PDF grounding, and SSE streaming) |
| `POST` | `/api/v1/documents/upload` | Upload dental textbook / PDF for parsing and embedding |
| `GET` | `/api/v1/documents` | List uploaded textbooks and processing status |
| `DELETE` | `/api/v1/documents/{id}` | Delete document and remove chunk vectors |
| `POST` | `/api/v1/study/quiz` | Generate dental examination MCQs with rationales |
| `POST` | `/api/v1/study/flashcards` | Generate active-recall flashcards on a dental topic |
| `GET` | `/api/v1/study/cards` | Retrieve saved flashcards and quiz questions |

---

## 📋 Example API Requests

### 1. Register a Student
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "dentist.alex@dental.edu",
    "password": "SecurePassword123!",
    "full_name": "Alex Mercer",
    "academic_level": "BDS Final Year"
  }'
```

### 2. Conversational Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_JWT_TOKEN>" \
  -d '{
    "message": "Explain the difference in nerve fiber involvement between reversible and irreversible pulpitis.",
    "stream": false
  }'
```

### 3. Generate Board Exam MCQs
```bash
curl -X POST "http://localhost:8000/api/v1/study/quiz" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_JWT_TOKEN>" \
  -d '{
    "topic": "Local Anesthetic Pharmacology & Toxicity",
    "num_questions": 3,
    "difficulty": "medium"
  }'
```

---

## ⚠️ Educational Disclaimer
The Dental AI Study & Research Assistant is strictly an educational tool designed for dental student instruction, revision, and academic literature discovery. It does **NOT** provide validated real-time clinical decision support or replace the licensed professional judgment of a qualified dentist.
