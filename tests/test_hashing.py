from investigation_rag.ingest.hashing import sha256_file, sha256_text


def test_sha256_file_matches_text_hash(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")

    assert sha256_file(path) == sha256_text("hello world")
