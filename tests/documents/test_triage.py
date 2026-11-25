import pymupdf
import pytest
from app.documents.triage import PageTriager, classify_page_metrics


def _make_text_pdf(tmp_path, pages_text):
    pdf_path = tmp_path / "text_book.pdf"
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def _make_image_pdf(tmp_path, include_caption=False):
    pdf_path = tmp_path / ("diagram.pdf" if include_caption else "scanned.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Create a solid raster pixmap covering >65% of page (or ~40% for diagram)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), False)
    pix.clear_with(128)
    if include_caption:
        rect = pymupdf.Rect(50, 50, 350, 450)  # ~24% area
        page.insert_image(rect, pixmap=pix)
        page.insert_text((50, 500), "Fig 4.1 Anatomy of the dental pulp chamber and root apex.", fontsize=11)
    else:
        rect = pymupdf.Rect(20, 20, 575, 822)  # ~88% area
        page.insert_image(rect, pixmap=pix)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def _make_table_pdf(tmp_path):
    pdf_path = tmp_path / "tables.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Draw a 3x3 grid table so PyMuPDF find_tables() detects it
    x0, y0 = 60, 100
    col_w, row_h = 140, 40
    for r in range(4):
        page.draw_line((x0, y0 + r * row_h), (x0 + 3 * col_w, y0 + r * row_h))
    for c in range(4):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + 3 * row_h))
    headers = ["Material", "Setting Time", "Compressive MPa"]
    rows = [
        ["MTA", "165 min", "40"],
        ["Biodentine", "12 min", "300"],
    ]
    for c, h in enumerate(headers):
        page.insert_text((x0 + c * col_w + 5, y0 + 25), h, fontsize=10)
    for r_idx, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * col_w + 5, y0 + r_idx * row_h + 25), val, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_triage_pure_text_pdf_routes_all_to_pymupdf(tmp_path):
    p1 = " ".join(["Endodontic instrumentation removes necrotic pulp tissue."] * 25)
    p2 = " ".join(["Sodium hypochlorite dissolves organic debris in root canals."] * 25)
    path = _make_text_pdf(tmp_path, [p1, p2])

    res = PageTriager.triage_pdf(path)
    assert res.total_pages == 2
    assert res.mean_words_per_page == pytest.approx(175.0, rel=0.01)
    assert all(p.suggested_extractor == "pymupdf" for p in res.pages)
    assert res.triage_summary["pymupdf"] == 2
    assert res.pages[0].text_coverage_pct > 0.0


def test_triage_scanned_pdf_routes_to_gemini(tmp_path):
    path = _make_image_pdf(tmp_path, include_caption=False)
    res = PageTriager.triage_pdf(path)

    assert res.total_pages == 1
    p = res.pages[0]
    assert p.word_count == 0
    assert p.image_coverage_pct > 60.0
    assert p.is_scanned is True
    assert p.suggested_extractor == "gemini"


def test_triage_table_heavy_pdf_routes_to_docling(tmp_path):
    path = _make_table_pdf(tmp_path)
    res = PageTriager.triage_pdf(path)

    assert res.total_pages == 1
    p = res.pages[0]
    assert p.table_count >= 1
    assert p.suggested_extractor == "docling"


def test_triage_mixed_dental_textbook_page():
    # Deterministic arithmetic verification for mixed page (table + diagram + prose)
    raw = {
        "page_number": 3,
        "word_count": 120,
        "char_count": 780,
        "table_count": 1,
        "image_count": 1,
        "image_coverage_pct": 35.0,
        "text_coverage_pct": 40.0,
    }
    pt = classify_page_metrics(raw, mean_words_per_page=250.0)
    assert pt.has_diagram is True
    assert pt.table_count == 1
    assert pt.suggested_extractor == "mixed"
    assert pt.mean_words_per_page == 250.0
