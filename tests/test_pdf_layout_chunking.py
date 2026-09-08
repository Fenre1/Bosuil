from pathlib import Path

import pytest

from investigation_rag.chunking.deterministic import DeterministicChunker
from investigation_rag.chunking.pdf_layout import PdfChunkingConfig, collect_layout_blocks, mark_recurring_page_furniture
from investigation_rag.extractors.pdf_extractor import PdfExtractor
from investigation_rag.extractors.base import ExtractedDocument, ExtractedSection
from investigation_rag.rag.citations import citation_label
from investigation_rag.retrieval.ranking import SearchResult


def test_pdf_chunking_removes_recurring_headers_and_allows_page_spans():
    document = pdf_document(
        [
            [
                block("Annual Report 2025 | Page 1", 1, 0.04, x0=0.12, x1=0.62, font_size=9),
                block("1 Risk Management", 1, 0.18, x0=0.12, x1=0.48, font_size=16, bold=True),
                block("Operational risk depends on clear ownership and", 1, 0.72),
                block("1", 1, 0.94, x0=0.48, x1=0.52, font_size=9),
            ],
            [
                block("Annual Report 2025 | Page 2", 2, 0.04, x0=0.12, x1=0.62, font_size=9),
                block("timely evidence preservation across investigations.", 2, 0.16),
                block("The team reviews every exception before closure.", 2, 0.19),
                block("2", 2, 0.94, x0=0.48, x1=0.52, font_size=9),
            ],
            [
                block("Annual Report 2025 | Page 3", 3, 0.04, x0=0.12, x1=0.62, font_size=9),
                block("Evidence owners record follow-up actions.", 3, 0.20),
                block("3", 3, 0.94, x0=0.48, x1=0.52, font_size=9),
            ],
        ],
        title="Annual Report 2025",
    )
    chunker = DeterministicChunker(target_chars=900, max_chars=1400, overlap_chars=0, pdf_soft_min_chars=0)

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert "Annual Report 2025 | Page" not in chunks[0].text
    assert "Operational risk depends on clear ownership and timely evidence preservation" in chunks[0].text
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 3
    assert chunks[0].metadata["pages"] == [1, 2, 3]
    assert chunks[0].metadata["heading_path_text"] == "1 Risk Management"
    assert chunks[0].metadata["source_spans"][0]["char_start"] == 0
    assert chunks[0].display_markdown.startswith("### 1 Risk Management")
    assert "Annual Report 2025 | Page" not in chunks[0].display_markdown


def test_pdf_headings_prevent_cross_section_packing():
    first = " ".join(["The first section keeps related risk context together."] * 7)
    second = " ".join(["The second section keeps findings separate from risk context."] * 7)
    document = pdf_document(
        [
            [
                block("1 Risk Management", 1, 0.12, font_size=16, bold=True),
                block(first, 1, 0.20),
                block("2 Findings", 1, 0.52, font_size=16, bold=True),
                block(second, 1, 0.60),
            ],
        ],
        title="Case Report",
    )
    chunker = DeterministicChunker(target_chars=900, max_chars=1400, overlap_chars=0, pdf_soft_min_chars=200)

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].metadata["heading_path_text"] == "1 Risk Management"
    assert chunks[1].metadata["heading_path_text"] == "2 Findings"
    assert "Section: 1 Risk Management" in chunks[0].metadata["embedding_context"]
    assert "Section: 2 Findings" in chunks[1].metadata["embedding_context"]
    assert "Section:" not in chunks[0].text
    assert chunks[0].display_markdown.startswith("### 1 Risk Management")
    assert chunks[1].display_markdown.startswith("### 2 Findings")


def test_pdf_display_markdown_formats_lists_and_tables():
    document = pdf_document(
        [
            [
                block("1 Findings", 1, 0.12, font_size=16, bold=True),
                block("The review found:", 1, 0.22),
                block("• Missing approval", 1, 0.26, content_type="list"),
                block("• Late escalation", 1, 0.29, content_type="list"),
                block("Control    Owner    Status", 1, 0.38, content_type="table"),
                block("Review     Alice    Open", 1, 0.41, content_type="table"),
            ],
        ],
        title="Case Report",
    )
    chunker = DeterministicChunker(target_chars=900, max_chars=1400, overlap_chars=0, pdf_soft_min_chars=0)

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].display_markdown == (
        "### 1 Findings\n\n"
        "The review found:\n\n"
        "- Missing approval\n"
        "- Late escalation"
    )
    assert chunks[1].display_markdown == (
        "```text\n"
        "Control    Owner    Status\n"
        "Review     Alice    Open\n"
        "```"
    )


def test_pdf_visual_figure_region_stays_out_of_following_heading():
    document = pdf_document(
        [
            [
                block("Figure 1. Screening flow", 1, 0.10, font_size=14, bold=True, reading_order=1),
                block("Results", 1, 0.75, font_size=16, bold=True, reading_order=2),
                block("The taxonomy paragraph belongs under the results heading.", 1, 0.82, font_size=11, reading_order=3),
                block("The flow diagram summarizes identification and screening.", 1, 0.16, font_size=11, reading_order=4),
                block("It reports included and excluded document counts.", 1, 0.19, font_size=11, reading_order=5),
                block("SYSTEMATIC SEARCH", 1, 0.34, x0=0.20, x1=0.45, font_size=8, bold=True, reading_order=6),
                block("ONGOING CONSULT", 1, 0.34, x0=0.56, x1=0.80, font_size=8, bold=True, reading_order=7),
                block("Records (n = 100)", 1, 0.43, x0=0.20, x1=0.40, font_size=8, reading_order=8),
                block("Screened (n = 80)", 1, 0.52, x0=0.20, x1=0.40, font_size=8, reading_order=9),
                block("Included (n = 40)", 1, 0.62, x0=0.45, x1=0.64, font_size=8, reading_order=10),
            ],
        ],
        title="Case Report",
    )
    chunker = DeterministicChunker(target_chars=900, max_chars=1400, overlap_chars=0, pdf_soft_min_chars=0)

    chunks = chunker.chunk_document(document)

    figure_chunks = [chunk for chunk in chunks if chunk.metadata.get("contains_figure")]
    assert figure_chunks
    assert figure_chunks[0].metadata["content_type"] == "figure"
    assert "**Figure**" in (figure_chunks[0].display_markdown or "")
    assert "```text" in (figure_chunks[0].display_markdown or "")
    assert "Screened (n = 80)" in figure_chunks[0].text

    results_chunks = [chunk for chunk in chunks if chunk.metadata.get("heading_path_text") == "Results"]
    assert results_chunks
    assert "Screened (n = 80)" not in results_chunks[0].text
    assert "The taxonomy paragraph belongs under the results heading." in results_chunks[0].text


def test_pdf_table_is_isolated_from_surrounding_prose():
    document = pdf_document(
        [
            [
                block("Findings", 1, 0.10, font_size=16, bold=True),
                block("The paragraph before the table explains the controls.", 1, 0.18),
                block("Control    Owner    Status", 1, 0.30, content_type="table"),
                block("Review     Alice    Open", 1, 0.34, content_type="table"),
                block("The paragraph after the table explains the outcome.", 1, 0.48),
            ]
        ],
        title="Case Report",
    )
    chunks = DeterministicChunker(
        target_chars=900,
        max_chars=1400,
        overlap_chars=240,
        pdf_soft_min_chars=1200,
    ).chunk_document(document)

    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("contains_table")]
    prose_chunks = [chunk for chunk in chunks if not chunk.metadata.get("contains_table")]

    assert len(table_chunks) == 1
    assert len(prose_chunks) == 2
    assert "paragraph before" not in table_chunks[0].text
    assert "paragraph after" not in table_chunks[0].text
    assert table_chunks[0].metadata["is_structural_chunk"] is True
    assert table_chunks[0].metadata["structural_isolation_reason"] == "isolated table region"
    assert table_chunks[0].metadata["overlap_text"] == ""


def test_pdf_figure_isolation_survives_soft_minimum_merging():
    document = pdf_document(
        [
            [
                block("Introductory prose belongs before the figure.", 1, 0.08, x1=0.68),
                block("Figure 1. Screening flow", 1, 0.18, font_size=14, bold=True),
                block("The diagram summarizes the screening process.", 1, 0.21),
                block("It reports the number retained at every stage.", 1, 0.24),
                block("SEARCH", 1, 0.26, x0=0.15, x1=0.30, font_size=8, bold=True),
                block("Records (n = 100)", 1, 0.36, x0=0.15, x1=0.35, font_size=8),
                block("Screened (n = 80)", 1, 0.47, x0=0.42, x1=0.61, font_size=8),
                block("Included (n = 40)", 1, 0.58, x0=0.66, x1=0.84, font_size=8),
                block("Closing prose.", 1, 0.76),
            ]
        ]
    )
    chunks = DeterministicChunker(
        target_chars=900,
        max_chars=1400,
        overlap_chars=240,
        pdf_soft_min_chars=1200,
    ).chunk_document(document)

    figure_chunks = [chunk for chunk in chunks if chunk.metadata.get("contains_figure")]

    assert len(figure_chunks) == 1
    assert "Introductory prose" not in figure_chunks[0].text
    assert "Closing prose" not in figure_chunks[0].text
    assert figure_chunks[0].metadata["structural_isolation_reason"] == "isolated figure region"


def test_recurring_furniture_detection_is_edge_limited():
    document = pdf_document(
        [
            [
                block("Recurring body warning", 1, 0.50),
                block("Case File | Page 1", 1, 0.04),
            ],
            [
                block("Recurring body warning", 2, 0.50),
                block("Case File | Page 2", 2, 0.04),
            ],
            [
                block("Recurring body warning", 3, 0.50),
                block("Case File | Page 3", 3, 0.04),
            ],
        ]
    )
    blocks = collect_layout_blocks(document)

    debug = mark_recurring_page_furniture(blocks, PdfChunkingConfig())

    removed_text = {item["text"] for item in debug["marked_blocks"]}
    assert "Case File | Page 1" in removed_text
    assert "Recurring body warning" not in removed_text


def test_pdf_chunker_version_changes_only_for_pdf_settings():
    default = DeterministicChunker()
    changed = DeterministicChunker(pdf_soft_min_chars=1400, pdf_overlap_mode="paragraph")

    assert default.version_for_source_type("txt") == "structured-v2"
    assert changed.version_for_source_type("txt") == "structured-v2"
    assert default.version_for_source_type("pdf") != changed.version_for_source_type("pdf")
    assert "soft_min=1400" in changed.version_for_source_type("pdf")
    assert "overlap=paragraph" in changed.version_for_source_type("pdf")


def test_page_range_citations_use_pdf_metadata():
    result = SearchResult(
        chunk_id="c1",
        investigation_id=1,
        file_id=1,
        path="C:/case/report.pdf",
        chunk_text="Evidence.",
        source_type="pdf",
        chunk_index=0,
        page_number=12,
        score=1.0,
        metadata={"page_start": 12, "page_end": 13, "pages": [12, 13]},
    )

    assert citation_label(1, result) == "[1] C:/case/report.pdf (pages 12-13)"


def test_real_1606_pdf_page_three_keeps_word_spaces():
    pytest.importorskip("fitz")
    path = Path(__file__).resolve().parents[2] / "test_pdf" / "1606.06565v2.pdf"
    if not path.exists():
        pytest.skip("shared PDF fixture is not available")

    document = PdfExtractor().extract(path)
    chunks = DeterministicChunker().chunk_document(document)
    page_three_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("page_start") == 3 or chunk.metadata.get("page_end") == 3 or 3 in chunk.metadata.get("pages", [])
    ]
    combined_text = "\n".join(chunk.text for chunk in page_three_chunks)
    combined_display = "\n".join(chunk.display_markdown or "" for chunk in page_three_chunks)

    assert "For concreteness, we will illustrate many of the accident risks" in combined_text
    assert "Forconcreteness,wewillillustrate" not in combined_text
    assert "Forconcreteness,wewillillustrate" not in combined_display


def test_real_2408_pdf_page_six_keeps_prisma_figure_separate_from_taxonomy_section():
    pytest.importorskip("fitz")
    path = Path(__file__).resolve().parents[2] / "test_pdf" / "2408.12622v3.pdf"
    if not path.exists():
        pytest.skip("shared PDF fixture is not available")

    document = PdfExtractor().extract(path)
    chunks = DeterministicChunker().chunk_document(document)
    page_six_chunks = [chunk for chunk in chunks if 6 in chunk.metadata.get("pages", [])]
    figure_chunks = [chunk for chunk in page_six_chunks if chunk.metadata.get("contains_figure")]
    taxonomy_chunks = [
        chunk
        for chunk in page_six_chunks
        if "Causal Taxonomy of AI Risks" in str(chunk.metadata.get("heading_path_text", ""))
    ]

    assert figure_chunks
    figure_text = "\n".join(chunk.text for chunk in figure_chunks)
    figure_display = "\n".join(chunk.display_markdown or "" for chunk in figure_chunks)
    assert "Figure 1. PRISMA Flow Diagram" in figure_text
    assert "SYSTEMATIC LITERATURE SEARCH" in figure_text
    assert "Total documents included (n = 74)" in figure_text
    assert "**Figure**" in figure_display
    assert "```text" in figure_display

    assert taxonomy_chunks
    taxonomy_text = "\n".join(chunk.text for chunk in taxonomy_chunks)
    assert "We created our taxonomies using a best-fit framework synthesis." in taxonomy_text
    assert "Screened title & abstract" not in taxonomy_text
    assert "SYSTEMATIC LITERATURE SEARCH" not in taxonomy_text
    assert "Records identified through database searching" not in taxonomy_text


def pdf_document(pages: list[list[dict]], title: str = "") -> ExtractedDocument:
    sections = [
        ExtractedSection(
            text="\n".join(item["text"] for item in page),
            source_type="pdf",
            page_number=index,
            metadata={"page_number": index, "layout_blocks": page},
        )
        for index, page in enumerate(pages, start=1)
    ]
    metadata = {"path": "C:/case/report.pdf", "page_count": len(pages)}
    if title:
        metadata["title"] = title
    return ExtractedDocument(source_type="pdf", sections=sections, metadata=metadata)


def block(
    text: str,
    page_number: int,
    y0_rel: float,
    *,
    x0: float = 0.12,
    x1: float = 0.88,
    font_size: float = 10.0,
    bold: bool = False,
    content_type: str = "",
    reading_order: int | None = None,
) -> dict:
    page_width = 600.0
    page_height = 800.0
    y1_rel = y0_rel + 0.018
    metadata = {"span_count": 1}
    if content_type:
        metadata["content_type"] = content_type
    return {
        "block_id": f"p{page_number}-{text[:12]}-{y0_rel}",
        "text": text,
        "page_number": page_number,
        "bbox": [x0 * page_width, y0_rel * page_height, x1 * page_width, y1_rel * page_height],
        "rel_bbox": [x0, y0_rel, x1, y1_rel],
        "page_width": page_width,
        "page_height": page_height,
        "font_size": font_size,
        "flags": 16 if bold else 0,
        "bold": bold,
        "italic": False,
        "reading_order": int(y0_rel * 1000) if reading_order is None else reading_order,
        "metadata": metadata,
    }
