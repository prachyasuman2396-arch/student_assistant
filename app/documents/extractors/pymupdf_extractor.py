import re
from typing import List, Optional, Tuple
import pymupdf
from app.documents.extractors.base import ExtractedChunk
from app.prompts.sanitizer import sanitize_untrusted_text


def _is_heading_block(text: str) -> bool:
    first_line = text.strip().split("\n")[0].strip()
    if not first_line or len(first_line) > 120:
        return False
    if re.match(r"^(chapter|section|module|unit|part)\s+\d+", first_line, re.IGNORECASE):
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", first_line):
        return True
    if first_line.isupper() and len(first_line.split()) <= 10:
        return True
    return False


class PyMuPDFExtractor:
    """
    Extracts structured prose passages from text-rich PDF pages using
    PyMuPDF `page.get_text("blocks", sort=True)` in reading order,
    grouping headings with following paragraphs and preserving bounding boxes.
    """

    @staticmethod
    def extract_page(
        file_path: str,
        doc_id: str,
        page_number: int,
        initial_heading_path: Optional[str] = None,
        exclude_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[ExtractedChunk]:
        doc = pymupdf.open(file_path)
        try:
            page = doc.load_page(page_number - 1)
            blocks = page.get_text("blocks", sort=True) or []
        finally:
            doc.close()

        chunks: List[ExtractedChunk] = []
        current_heading = initial_heading_path or f"Page {page_number}"
        passage_parts: List[str] = []
        passage_bbox: Optional[List[float]] = None

        def _overlaps_excluded(bbox: Tuple[float, float, float, float]) -> bool:
            if not exclude_bboxes:
                return False
            x0, y0, x1, y1 = bbox
            for ex0, ey0, ex1, ey1 in exclude_bboxes:
                ix0, iy0 = max(x0, ex0), max(y0, ey0)
                ix1, iy1 = min(x1, ex1), min(y1, ey1)
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    area = max(1e-6, (x1 - x0) * (y1 - y0))
                    if inter / area > 0.5:
                        return True
            return False

        def _flush_passage():
            nonlocal passage_parts, passage_bbox
            if not passage_parts:
                return
            combined = sanitize_untrusted_text("\n\n".join(passage_parts))
            if combined:
                chunks.append(
                    ExtractedChunk(
                        doc_id=doc_id,
                        page_number=page_number,
                        bbox=tuple(passage_bbox) if passage_bbox else None,  # type: ignore
                        chunk_type="text",
                        content=combined,
                        extractor="pymupdf",
                        heading_path=current_heading,
                        extraction_meta={"char_count": len(combined)},
                    )
                )
            passage_parts = []
            passage_bbox = None

        for blk in blocks:
            if len(blk) < 7 or blk[6] != 0:  # skip non-text blocks
                continue
            bbox = (float(blk[0]), float(blk[1]), float(blk[2]), float(blk[3]))
            if _overlaps_excluded(bbox):
                continue

            raw_text = str(blk[4]).strip()
            if not raw_text:
                continue

            if _is_heading_block(raw_text):
                _flush_passage()
                heading_line = raw_text.split("\n")[0].strip()[:120]
                current_heading = heading_line
                passage_parts.append(raw_text)
                passage_bbox = list(bbox)
            else:
                passage_parts.append(raw_text)
                if passage_bbox is None:
                    passage_bbox = list(bbox)
                else:
                    passage_bbox[0] = min(passage_bbox[0], bbox[0])
                    passage_bbox[1] = min(passage_bbox[1], bbox[1])
                    passage_bbox[2] = max(passage_bbox[2], bbox[2])
                    passage_bbox[3] = max(passage_bbox[3], bbox[3])

        _flush_passage()
        return chunks
