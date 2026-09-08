from investigation_rag.chunking.deterministic import DeterministicChunker
from investigation_rag.extractors.base import ExtractedDocument, ExtractedSection


def test_chunking_preserves_page_metadata():
    document = ExtractedDocument(
        source_type="pdf",
        sections=[
            ExtractedSection(
                text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
                source_type="pdf",
                page_number=7,
            )
        ],
    )
    chunker = DeterministicChunker(target_chars=24, max_chars=80, overlap_chars=0)

    chunks = chunker.chunk_document(document)

    assert len(chunks) >= 2
    assert {chunk.page_number for chunk in chunks} == {7}
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_legacy_extraction_uses_source_text_when_no_formatting_mapping_exists():
    document = ExtractedDocument(
        source_type="docx",
        sections=[
            ExtractedSection(
                text="Heading\n\nFirst paragraph.",
                source_type="docx",
                display_markdown="# Heading\n\nFirst paragraph.",
            )
        ],
    )
    chunker = DeterministicChunker(target_chars=400, max_chars=800, overlap_chars=0)

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == "Heading\n\nFirst paragraph."
    assert chunks[0].metadata["display_html"] == "<p>Heading</p><p>First paragraph.</p>"
    assert chunks[0].display_markdown is None


def test_chunking_never_guesses_formatted_passages_from_lengths():
    document = ExtractedDocument(
        source_type="pdf",
        sections=[
            ExtractedSection(
                text="Intro paragraph.\n\nHeading\nDetails about the heading. More details follow.",
                source_type="pdf",
                display_markdown="Intro paragraph.\n\n### Heading\n\nDetails about the heading. More details follow.",
            )
        ],
    )
    chunker = DeterministicChunker(target_chars=20, max_chars=120, overlap_chars=0)

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].metadata["display_html"] == "<p>Intro paragraph.</p>"
    assert "Intro paragraph" not in chunks[1].metadata["display_html"]
    assert "Heading<br>Details about the heading." in chunks[1].metadata["display_html"]
