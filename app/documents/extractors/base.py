import uuid
from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class ExtractedChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    page_number: int = Field(..., ge=1)
    bbox: Optional[Tuple[float, float, float, float]] = None
    chunk_type: Literal["text", "table", "figure_caption", "figure_description"] = "text"
    content: str
    extractor: Literal["pymupdf", "gemini", "docling"]
    heading_path: Optional[str] = None
    extraction_meta: Dict[str, Any] = Field(default_factory=dict)
