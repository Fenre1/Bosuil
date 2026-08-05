import json
from pathlib import Path

from investigation_rag.ingest.pipeline import ScanPipeline
from investigation_rag.ingest.searchable_text import keyword_searchable_text, stable_embedding_text


def test_index_inputs_are_move_stable_because_path_has_its_own_fts_column():
    path = Path("C:/Evidence/Invoices/payment.txt")
    metadata = {"subject": "Payment", "sender": "Alice"}

    keyword = keyword_searchable_text(path, "Paid on Tuesday", metadata)
    embedding = stable_embedding_text("Paid on Tuesday", metadata)

    assert str(path) not in keyword
    assert path.name not in keyword
    assert "Evidence" not in embedding
    assert path.name not in embedding
    assert "Subject: Payment" in embedding
    assert "Paid on Tuesday" in embedding


def test_embedding_context_deduplicates_equivalent_fields():
    embedding = stable_embedding_text("Body", {"subject": "Same", "email_subject": "Same"})

    assert embedding.count("Subject: Same") == 1


def test_attachment_embedding_input_is_identical_when_reloaded_from_sqlite_metadata():
    metadata = {
        "attachment_name": "appendix.txt",
        "parent_email_subject": "Case update",
        "parent_email_sender": "Alice",
    }
    initial = stable_embedding_text("Attachment evidence", metadata)
    row = {
        "metadata_json": json.dumps(metadata),
        "email_subject": "Case update",
        "email_sender": "Alice",
        "attachment_name": "appendix.txt",
    }

    reloaded = stable_embedding_text(
        "Attachment evidence",
        ScanPipeline._stored_chunk_metadata(row),
    )

    assert reloaded == initial
