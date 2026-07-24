from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.ingest.hashing import sha256_text


@dataclass(frozen=True)
class EmbeddingTiming:
    cache_lookup_seconds: float = 0.0
    ollama_seconds: float = 0.0
    cache_write_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    ollama_requests: int = 0


class EmbeddingCache:
    def __init__(self, db: SQLiteDB, ollama: OllamaClient, model: str):
        self.db = db
        self.ollama = ollama
        self.model = model
        self.last_timing = EmbeddingTiming()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        lookup_seconds = 0.0
        write_seconds = 0.0
        ollama_seconds = 0.0
        cache_hits = 0
        vectors: list[list[float] | None] = []
        missing_texts: list[str] = []
        missing_hashes: list[str] = []

        text_hashes = [sha256_text(text) for text in texts]
        started = perf_counter()
        cached_by_hash = self.db.get_embeddings(text_hashes, self.model)
        lookup_seconds = perf_counter() - started
        for text, text_hash in zip(texts, text_hashes):
            cached = cached_by_hash.get(text_hash)
            if cached is None:
                vectors.append(None)
                missing_texts.append(text)
                missing_hashes.append(text_hash)
            else:
                cache_hits += 1
                vectors.append(cached)

        if missing_texts:
            requests_before = getattr(self.ollama, "request_count", 0)
            started = perf_counter()
            try:
                generated = self.ollama.embed(self.model, missing_texts)
            except Exception:
                ollama_seconds = perf_counter() - started
                requests_after = getattr(self.ollama, "request_count", requests_before + 1)
                self.last_timing = EmbeddingTiming(
                    cache_lookup_seconds=lookup_seconds,
                    ollama_seconds=ollama_seconds,
                    cache_hits=cache_hits,
                    cache_misses=len(missing_texts),
                    ollama_requests=max(1, requests_after - requests_before),
                )
                raise
            ollama_seconds = perf_counter() - started
            requests_after = getattr(self.ollama, "request_count", requests_before + 1)
            if len(generated) != len(missing_texts):
                raise RuntimeError("Ollama returned an unexpected number of embeddings.")
            generated_iter = iter(generated)
            for index, vector in enumerate(vectors):
                if vector is None:
                    new_vector = next(generated_iter)
                    vectors[index] = new_vector
            started = perf_counter()
            self.db.set_embeddings(list(zip(missing_hashes, generated)), self.model)
            write_seconds = perf_counter() - started

        self.last_timing = EmbeddingTiming(
            cache_lookup_seconds=lookup_seconds,
            ollama_seconds=ollama_seconds,
            cache_write_seconds=write_seconds,
            cache_hits=cache_hits,
            cache_misses=len(missing_texts),
            ollama_requests=max(1, requests_after - requests_before) if missing_texts else 0,
        )

        return [vector for vector in vectors if vector is not None]
