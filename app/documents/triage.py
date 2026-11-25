from typing import List, Dict, Any, Literal, Optional, Tuple
import pymupdf
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import logger
from app.core.observability import traceable


class PageTriage(BaseModel):
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    word_count: int = Field(..., ge=0)
    char_count: int = Field(..., ge=0)
    table_count: int = Field(..., ge=0)
    image_count: int = Field(..., ge=0)
    image_coverage_pct: float = Field(..., ge=0.0, le=100.0)
    text_coverage_pct: float = Field(..., ge=0.0, le=100.0)
    is_scanned: bool = False
    has_diagram: bool = False
    suggested_extractor: Literal["pymupdf", "gemini", "docling", "mixed"]
    mean_words_per_page: float = Field(0.0, ge=0.0)
    # Optional region hints for mixed pages
    table_bboxes: List[Tuple[float, float, float, float]] = Field(default_factory=list)
    image_bboxes: List[Tuple[float, float, float, float]] = Field(default_factory=list)


class DocumentTriageResult(BaseModel):
    total_pages: int
    mean_words_per_page: float
    pages: List[PageTriage]
    triage_summary: Dict[str, int] = Field(
        default_factory=lambda: {"pymupdf": 0, "gemini": 0, "docling": 0, "mixed": 0}
    )


def _rect_area(rect: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def classify_page_metrics(
    raw_metrics: Dict[str, Any],
    mean_words_per_page: float,
    word_floor: int = settings.TRIAGE_SCANNED_WORD_FLOOR,
    word_ratio: float = settings.TRIAGE_SCANNED_WORD_RATIO,
    scanned_img_pct: float = settings.TRIAGE_SCANNED_IMAGE_COVERAGE_PCT,
) -> PageTriage:
    """
    Pure arithmetic classifier applying document-aware triage heuristics to raw page metrics.
    Separated for deterministic unit testing.
    """
    word_count = raw_metrics["word_count"]
    table_count = raw_metrics["table_count"]
    image_count = raw_metrics["image_count"]
    image_coverage_pct = raw_metrics["image_coverage_pct"]
    drawing_count = raw_metrics.get("drawing_count", 0)

    is_scanned = (
        word_count < max(word_floor, word_ratio * mean_words_per_page)
        and image_coverage_pct > scanned_img_pct
    )

    # Diagram heuristic: illustration with light caption OR high vector drawing density
    has_raster_diagram = (
        image_count >= 1
        and 20.0 < image_coverage_pct < 80.0
        and word_count < mean_words_per_page
    )
    has_vector_diagram = (
        drawing_count >= 25
        and word_count < max(word_floor, 0.5 * mean_words_per_page)
    )
    has_diagram = bool(has_raster_diagram or has_vector_diagram)

    # Routing decision per page
    if table_count > 0:
        # If page also has substantial prose + diagram, mark as mixed; otherwise docling
        if (is_scanned or has_diagram) and word_count >= 30:
            suggested = "mixed"
        else:
            suggested = "docling"
    elif is_scanned or has_diagram:
        suggested = "gemini"
    elif mean_words_per_page > 0 and word_count >= 0.6 * mean_words_per_page:
        suggested = "pymupdf"
    elif mean_words_per_page == 0 and word_count == 0:
        # Completely blank or unmeasured: default to gemini if images exist else pymupdf
        suggested = "gemini" if image_count > 0 else "pymupdf"
    else:
        suggested = "mixed"

    return PageTriage(
        page_number=raw_metrics["page_number"],
        word_count=word_count,
        char_count=raw_metrics["char_count"],
        table_count=table_count,
        image_count=image_count,
        image_coverage_pct=round(image_coverage_pct, 2),
        text_coverage_pct=round(raw_metrics["text_coverage_pct"], 2),
        is_scanned=is_scanned,
        has_diagram=has_diagram,
        suggested_extractor=suggested,
        mean_words_per_page=round(mean_words_per_page, 2),
        table_bboxes=raw_metrics.get("table_bboxes", []),
        image_bboxes=raw_metrics.get("image_bboxes", []),
    )


class PageTriager:
    """
    Runs a single PyMuPDF pass over an uploaded PDF to collect per-page layout
    statistics, computes document-wide `mean_words_per_page`, and routes each page
    to the optimal extractor ('pymupdf', 'gemini', 'docling', or 'mixed').
    """

    @staticmethod
    @traceable(name="document_triage", run_type="tool")
    def triage_pdf(file_path: str) -> DocumentTriageResult:
        doc = pymupdf.open(file_path)
        raw_pages: List[Dict[str, Any]] = []
        total_words = 0

        try:
            for idx in range(doc.page_count):
                page = doc.load_page(idx)
                page_area = max(1.0, float(page.rect.width * page.rect.height))

                raw_text = page.get_text("text") or ""
                words = raw_text.split()
                word_count = len(words)
                char_count = len(raw_text.strip())
                total_words += word_count

                # Text coverage from blocks
                text_area = 0.0
                for blk in page.get_text("blocks") or []:
                    if len(blk) >= 7 and blk[6] == 0:  # text block
                        text_area += _rect_area((blk[0], blk[1], blk[2], blk[3]))
                text_coverage_pct = min(100.0, (text_area / page_area) * 100.0)

                # Image count & bbox coverage
                images = page.get_images(full=True) or []
                image_count = len(images)
                image_area = 0.0
                image_bboxes: List[Tuple[float, float, float, float]] = []
                for img in images:
                    xref = img[0]
                    try:
                        rects = page.get_image_rects(xref)
                        for r in rects:
                            bbox = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                            image_bboxes.append(bbox)
                            image_area += _rect_area(bbox)
                    except Exception:
                        pass
                image_coverage_pct = min(100.0, (image_area / page_area) * 100.0)

                # Table count & bboxes
                table_count = 0
                table_bboxes: List[Tuple[float, float, float, float]] = []
                try:
                    tabs = page.find_tables()
                    if tabs and hasattr(tabs, "tables"):
                        table_count = len(tabs.tables)
                        for t in tabs.tables:
                            if hasattr(t, "bbox") and t.bbox:
                                table_bboxes.append(tuple(float(c) for c in t.bbox[:4]))
                except Exception as te:
                    logger.debug(f"find_tables failed on page {idx + 1}: {te}")

                # Vector drawings density heuristic
                drawing_count = 0
                try:
                    drawing_count = len(page.get_drawings() or [])
                except Exception:
                    pass

                raw_pages.append({
                    "page_number": idx + 1,
                    "word_count": word_count,
                    "char_count": char_count,
                    "table_count": table_count,
                    "image_count": image_count,
                    "image_coverage_pct": image_coverage_pct,
                    "text_coverage_pct": text_coverage_pct,
                    "drawing_count": drawing_count,
                    "table_bboxes": table_bboxes,
                    "image_bboxes": image_bboxes,
                })
        finally:
            doc.close()

        total_pages = len(raw_pages)
        mean_words = (total_words / total_pages) if total_pages > 0 else 0.0

        triaged_pages: List[PageTriage] = []
        summary = {"pymupdf": 0, "gemini": 0, "docling": 0, "mixed": 0}

        for raw in raw_pages:
            pt = classify_page_metrics(raw, mean_words_per_page=mean_words)
            triaged_pages.append(pt)
            summary[pt.suggested_extractor] += 1

        logger.info(
            f"Triage complete for '{file_path}': {total_pages} pages, "
            f"mean_words={mean_words:.1f}, summary={summary}"
        )

        return DocumentTriageResult(
            total_pages=total_pages,
            mean_words_per_page=round(mean_words, 2),
            pages=triaged_pages,
            triage_summary=summary,
        )
