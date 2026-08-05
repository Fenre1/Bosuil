from investigation_rag.chunking.deterministic import DeterministicChunker
from investigation_rag.ingest.pipeline_state import pipeline_signatures


def test_non_pdf_chunk_signature_includes_all_size_settings():
    baseline = pipeline_signatures(".txt", DeterministicChunker(), "embed")
    changed_target = pipeline_signatures(".txt", DeterministicChunker(target_chars=3000), "embed")
    changed_overlap = pipeline_signatures(".txt", DeterministicChunker(overlap_chars=100), "embed")

    assert baseline.chunking != changed_target.chunking
    assert baseline.chunking != changed_overlap.chunking
    assert baseline.extraction == changed_target.extraction


def test_stage_signatures_keep_embedding_model_independent():
    chunker = DeterministicChunker()
    first = pipeline_signatures("pdf", chunker, "model-a")
    second = pipeline_signatures(".pdf", chunker, "model-b")

    assert first.extraction == second.extraction
    assert first.chunking == second.chunking
    assert first.metadata == second.metadata
    assert first.embedding_input == second.embedding_input
    assert first.embedding_model != second.embedding_model
