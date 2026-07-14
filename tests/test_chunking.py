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


def test_chunking_carries_matching_display_markdown():
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
    assert chunks[0].display_markdown == "# Heading\n\nFirst paragraph."


def test_chunking_fuzzily_packs_display_markdown_when_paragraphs_differ():
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
    assert chunks[0].display_markdown == "Intro paragraph."
    assert chunks[1].display_markdown == "### Heading\n\nDetails about the heading. More details follow."
