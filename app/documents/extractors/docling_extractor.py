from typing import List, Optional, Tuple
from app.core.logging import logger
from app.core.observability import traceable
from app.documents.extractors.base import ExtractedChunk
from app.prompts.sanitizer import sanitize_untrusted_text


class DoclingTableExtractor:
    """
    Extracts structured tables as GitHub-flavored markdown from PDF pages.
    Fallback chain: Docling -> Marker -> Unstructured partition_pdf.
    Logs which path ran; never silently falls back to PyMuPDF.
    """

    def _run_docling(self, file_path: str, page_number: int) -> List[Tuple[str, Optional[str]]]:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path, page_range=(page_number, page_number))
        tables_out: List[Tuple[str, Optional[str]]] = []
        doc = result.document
        for tbl in getattr(doc, "tables", []):
            md = tbl.export_to_markdown() if hasattr(tbl, "export_to_markdown") else str(tbl)
            html = tbl.export_to_html() if hasattr(tbl, "export_to_html") else None
            if md and md.strip():
                tables_out.append((md.strip(), html))
        return tables_out

    def _run_marker(self, file_path: str, page_number: int) -> List[Tuple[str, Optional[str]]]:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(file_path)
        md_text = getattr(rendered, "markdown", str(rendered))
        tables_out: List[Tuple[str, Optional[str]]] = []
        # Extract markdown table blocks
        current_tbl: List[str] = []
        for line in md_text.splitlines():
            if line.strip().startswith("|"):
                current_tbl.append(line)
            elif current_tbl:
                tables_out.append(("\n".join(current_tbl), None))
                current_tbl = []
        if current_tbl:
            tables_out.append(("\n".join(current_tbl), None))
        return tables_out

    def _run_unstructured(self, file_path: str, page_number: int) -> List[Tuple[str, Optional[str]]]:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
        )
        tables_out: List[Tuple[str, Optional[str]]] = []
        for el in elements:
            el_page = getattr(getattr(el, "metadata", None), "page_number", None)
            if el_page is not None and el_page != page_number:
                continue
            if getattr(el, "category", "") == "Table":
                html = getattr(getattr(el, "metadata", None), "text_as_html", None)
                text = str(el).strip()
                if text:
                    tables_out.append((text, html))
        return tables_out

    @traceable(name="docling_table_extract", run_type="tool")
    def extract_page(
        self,
        file_path: str,
        doc_id: str,
        page_number: int,
        table_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[ExtractedChunk]:
        tables_raw: List[Tuple[str, Optional[str]]] = []
        used_path = "docling"

        try:
            tables_raw = self._run_docling(file_path, page_number)
            logger.info(f"[TableExtractor] Page {page_number}: extracted {len(tables_raw)} tables via docling.")
        except Exception as doc_err:
            logger.warning(f"[TableExtractor] Docling failed on page {page_number} ({doc_err}); trying Marker.")
            used_path = "marker"
            try:
                tables_raw = self._run_marker(file_path, page_number)
                logger.info(f"[TableExtractor] Page {page_number}: extracted {len(tables_raw)} tables via marker.")
            except Exception as m_err:
                logger.warning(f"[TableExtractor] Marker failed on page {page_number} ({m_err}); trying Unstructured.")
                used_path = "unstructured"
                try:
                    tables_raw = self._run_unstructured(file_path, page_number)
                    logger.info(
                        f"[TableExtractor] Page {page_number}: extracted {len(tables_raw)} tables via unstructured."
                    )
                except Exception as u_err:
                    logger.error(
                        f"[TableExtractor] All table extractors (docling, marker, unstructured) failed on page {page_number}: {u_err}"
                    )
                    raise RuntimeError(
                        f"DoclingTableExtractor failed on page {page_number}: docling={doc_err}; marker={m_err}; unstructured={u_err}"
                    )

        chunks: List[ExtractedChunk] = []
        for idx, (md_content, html_content) in enumerate(tables_raw):
            bbox = table_bboxes[idx] if (table_bboxes and idx < len(table_bboxes)) else None
            sanitized = sanitize_untrusted_text(md_content)
            if not sanitized:
                continue
            chunks.append(
                ExtractedChunk(
                    doc_id=doc_id,
                    page_number=page_number,
                    bbox=bbox,
                    chunk_type="table",
                    content=sanitized,
                    extractor="docling",
                    extraction_meta={
                        "backend_path": used_path,
                        "table_index": idx,
                        "table_html": html_content,
                    },
                )
            )

        return chunks
