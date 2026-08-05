from pathlib import Path

from investigation_rag.ingest.pipeline import classify_manifest_change
from investigation_rag.ingest.scanner import ScannedFile


def scanned_file(size=10, modified_ns=123) -> ScannedFile:
    return ScannedFile(
        path=Path("C:/case/file.txt"),
        normalized_path="c:/case/file.txt",
        suffix=".txt",
        stable_id="1:2",
        size=size,
        modified_ns=modified_ns,
    )


def test_new_file_needs_hash_and_extraction():
    decision = classify_manifest_change(None, scanned_file())

    assert decision.reason == "new"
    assert decision.needs_hash is True
    assert decision.needs_extraction is True


def test_unchanged_metadata_skips_hash_and_extraction():
    existing = {"size": 10, "modified_ns": 123, "content_hash": "abc"}

    decision = classify_manifest_change(existing, scanned_file())

    assert decision.reason == "unchanged_metadata"
    assert decision.needs_hash is False
    assert decision.needs_extraction is False


def test_modified_file_is_suspicious():
    existing = {"size": 10, "modified_ns": 122, "content_hash": "abc"}

    decision = classify_manifest_change(existing, scanned_file())

    assert decision.reason == "suspicious_or_changed"
    assert decision.needs_hash is True


def test_replaced_file_identity_is_suspicious_even_when_size_and_time_match():
    existing = {
        "size": 10,
        "modified_ns": 123,
        "stable_id": "1:99",
        "content_hash": "abc",
    }

    decision = classify_manifest_change(existing, scanned_file())

    assert decision.reason == "suspicious_or_changed"
    assert decision.needs_hash is True
