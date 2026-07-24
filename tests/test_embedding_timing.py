from pathlib import Path

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.embedding_cache import EmbeddingCache


class FakeOllama:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.calls.append((model, texts))
        return [[float(index), 1.0] for index, _text in enumerate(texts)]


def test_embedding_cache_reports_hits_misses_and_request_timing(tmp_path: Path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    ollama = FakeOllama()
    cache = EmbeddingCache(db, ollama, "test-model")

    cache.embed_texts(["alpha", "beta"])
    first = cache.last_timing

    assert first.cache_hits == 0
    assert first.cache_misses == 2
    assert first.ollama_requests == 1
    assert first.cache_lookup_seconds >= 0
    assert first.cache_write_seconds >= 0
    assert first.ollama_seconds >= 0

    cache.embed_texts(["alpha", "beta"])
    second = cache.last_timing

    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert second.ollama_requests == 0
    assert len(ollama.calls) == 1
