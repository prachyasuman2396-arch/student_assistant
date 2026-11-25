import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.documents.triage import DocumentTriageResult, PageTriage
from app.documents.extractors.base import ExtractedChunk
from app.documents.extractors.pymupdf_extractor import PyMuPDFExtractor
from app.documents.extractors.gemini_extractor import GeminiVisionExtractor
from app.documents.extractors.docling_extractor import DoclingTableExtractor

_EXTRACTION_EXECUTOR = ThreadPoolExecutor(max_workers=settings.EXTRACTION_THREAD_POOL_SIZE)


class ExtractionCoordinator:
    """
    Routes each triaged page to its designated extractor ('pymupdf', 'gemini', 'docling', or 'mixed'),
    running sync PDF/Docling calls on a bounded ThreadPoolExecutor and bounding page concurrency.
    """

    def __init__(
        self,
        pymupdf_extractor: Optional[PyMuPDFExtractor] = None,
        gemini_extractor: Optional[GeminiVisionExtractor] = None,
        docling_extractor: Optional[DoclingTableExtractor] = None,
    ):
        self.pymupdf = pymupdf_extractor or PyMuPDFExtractor()
        self.gemini = gemini_extractor or GeminiVisionExtractor()
        self.docling = docling_extractor or DoclingTableExtractor()

    def _extract_single_page_sync(
        self,
        file_path: str,
        doc_id: str,
        page_triage: PageTriage,
    ) -> List[ExtractedChunk]:
        ext = page_triage.suggested_extractor
        pnum = page_triage.page_number

        if ext == "pymupdf":
            return self.pymupdf.extract_page(file_path, doc_id, pnum)

        if ext == "gemini":
            return self.gemini.extract_page(file_path, doc_id, pnum)

        if ext == "docling":
            return self.docling.extract_page(
                file_path, doc_id, pnum, table_bboxes=page_triage.table_bboxes
            )

        # "mixed" page: split text blocks, tables, and figures, then merge in reading order
        merged: List[ExtractedChunk] = []
        excluded = list(page_triage.table_bboxes) + list(page_triage.image_bboxes)

        merged.extend(
            self.pymupdf.extract_page(file_path, doc_id, pnum, exclude_bboxes=excluded)
        )

        if page_triage.table_count > 0:
            try:
                merged.extend(
                    self.docling.extract_page(
                        file_path, doc_id, pnum, table_bboxes=page_triage.table_bboxes
                    )
                )
            except Exception as e:
                logger.error(f"Mixed page {pnum} table extraction failed: {e}")

        if page_triage.image_count > 0:
            if page_triage.image_bboxes:
                for img_bbox in page_triage.image_bboxes:
                    try:
                        merged.extend(
                            self.gemini.extract_page(file_path, doc_id, pnum, crop_bbox=img_bbox)
                        )
                    except Exception as e:
                        logger.error(f"Mixed page {pnum} figure extraction failed: {e}")
            else:
                try:
                    merged.extend(self.gemini.extract_page(file_path, doc_id, pnum))
                except Exception as e:
                    logger.error(f"Mixed page {pnum} figure extraction failed: {e}")

        # Sort in reading order by vertical y0 coordinate (fallback to 0.0)
        merged.sort(key=lambda c: (c.bbox[1] if c.bbox else 0.0, c.bbox[0] if c.bbox else 0.0))
        return merged

    async def extract_document(
        self,
        file_path: str,
        doc_id: str,
        triage_result: DocumentTriageResult,
    ) -> List[ExtractedChunk]:
        semaphore = asyncio.Semaphore(settings.EXTRACTION_CONCURRENCY)
        loop = asyncio.get_running_loop()

        async def _run_page(pt: PageTriage) -> List[ExtractedChunk]:
            async with semaphore:
                return await loop.run_in_executor(
                    _EXTRACTION_EXECUTOR,
                    self._extract_single_page_sync,
                    file_path,
                    doc_id,
                    pt,
                )

        page_results = await asyncio.gather(*[_run_page(pt) for pt in triage_result.pages])
        all_chunks: List[ExtractedChunk] = []
        for page_chunks in page_results:
            all_chunks.extend(page_chunks)
        return all_chunks


__all__ = [
    "ExtractedChunk",
    "PyMuPDFExtractor",
    "GeminiVisionExtractor",
    "DoclingTableExtractor",
    "ExtractionCoordinator",
]
