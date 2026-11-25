import base64
import json
from typing import List, Literal, Optional, Tuple
import pymupdf
from pydantic import BaseModel, Field, ValidationError
from app.core.config import settings
from app.core.logging import logger
from app.core.observability import traceable
from app.documents.extractors.base import ExtractedChunk
from app.prompts.sanitizer import sanitize_untrusted_text

VISION_EXTRACTION_PROMPT = (
    "You are a dental textbook figure reader. Output structured markdown with: "
    "(1) Figure type (anatomy diagram / flowchart / chart / photo / radiograph / scanned text). "
    "(2) If scanned text, transcribe verbatim. "
    "(3) If diagram, describe anatomy, labels, arrows, regions. "
    "(4) Any visible caption."
)


class VisionFigureSchema(BaseModel):
    figure_type: Literal[
        "anatomy diagram",
        "flowchart",
        "chart",
        "photo",
        "radiograph",
        "scanned text",
    ]
    transcription_or_description: str = Field(..., min_length=1)
    labels_and_regions: List[str] = Field(default_factory=list)
    visible_caption: Optional[str] = None


class GeminiVisionExtractor:
    """
    Rasterizes PDF pages (or cropped figure bounding boxes) to PNG at 200 DPI
    and extracts structured descriptions/transcriptions via Gemini Vision with
    JSON schema validation and bounded retries.
    Never silently falls back to PyMuPDF on failure.
    """

    def __init__(self, vision_client=None, model: Optional[str] = None):
        self._vision_client = vision_client
        self.model = model or settings.GEMINI_VISION_MODEL

    def _default_vision_call(self, image_bytes: bytes, prompt: str) -> str:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured for GeminiVisionExtractor.")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=base64.b64decode(b64_img), mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VisionFigureSchema,
                temperature=0.1,
            ),
        )
        return response.text or ""

    @staticmethod
    def rasterize_page(
        file_path: str,
        page_number: int,
        dpi: int = settings.PYMUPDF_DPI,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bytes:
        doc = pymupdf.open(file_path)
        try:
            page = doc.load_page(page_number - 1)
            clip = pymupdf.Rect(*crop_bbox) if crop_bbox else None
            pixmap = page.get_pixmap(dpi=dpi, clip=clip)
            return pixmap.tobytes("png")
        finally:
            doc.close()

    @traceable(name="gemini_vision_extract", run_type="llm")
    def extract_page(
        self,
        file_path: str,
        doc_id: str,
        page_number: int,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        dpi: int = settings.PYMUPDF_DPI,
    ) -> List[ExtractedChunk]:
        png_bytes = self.rasterize_page(
            file_path=file_path,
            page_number=page_number,
            dpi=dpi,
            crop_bbox=crop_bbox,
        )

        last_err: Optional[Exception] = None
        max_attempts = settings.GEMINI_VISION_MAX_RETRIES + 1
        parsed: Optional[VisionFigureSchema] = None

        for attempt in range(1, max_attempts + 1):
            try:
                if self._vision_client is not None:
                    raw_json = self._vision_client(png_bytes, VISION_EXTRACTION_PROMPT)
                else:
                    raw_json = self._default_vision_call(png_bytes, VISION_EXTRACTION_PROMPT)

                data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
                parsed = VisionFigureSchema.model_validate(data)
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as ve:
                last_err = ve
                logger.warning(
                    f"[GeminiVision] Schema validation failed on page {page_number} "
                    f"(attempt {attempt}/{max_attempts}): {ve}"
                )
            except Exception as exc:
                last_err = exc
                logger.error(
                    f"[GeminiVision] Vision API error on page {page_number} "
                    f"(attempt {attempt}/{max_attempts}): {exc}"
                )

        if parsed is None:
            raise RuntimeError(
                f"GeminiVisionExtractor failed for page {page_number} after {max_attempts} attempts: {last_err}"
            )

        sanitized_desc = sanitize_untrusted_text(parsed.transcription_or_description)
        labels_str = ", ".join(sanitize_untrusted_text(lbl) for lbl in parsed.labels_and_regions if lbl)

        body_parts = [
            f"Figure Type: {parsed.figure_type}",
            sanitized_desc,
        ]
        if labels_str:
            body_parts.append(f"Labels & Regions: {labels_str}")

        chunks: List[ExtractedChunk] = [
            ExtractedChunk(
                doc_id=doc_id,
                page_number=page_number,
                bbox=crop_bbox,
                chunk_type="figure_description",
                content="\n".join(body_parts),
                extractor="gemini",
                extraction_meta={
                    "model": self.model,
                    "dpi": dpi,
                    "figure_type": parsed.figure_type,
                },
            )
        ]

        if parsed.visible_caption:
            sanitized_caption = sanitize_untrusted_text(parsed.visible_caption)
            if sanitized_caption:
                chunks.append(
                    ExtractedChunk(
                        doc_id=doc_id,
                        page_number=page_number,
                        bbox=crop_bbox,
                        chunk_type="figure_caption",
                        content=sanitized_caption,
                        extractor="gemini",
                        extraction_meta={"model": self.model, "dpi": dpi},
                    )
                )

        return chunks
