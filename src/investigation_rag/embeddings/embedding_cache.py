from __future__ import annotations

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.ingest.hashing import sha256_text


class EmbeddingCache:
    def __init__(self, db: SQLiteDB, ollama: OllamaClient, model: str):
        self.db = db
        self.ollama = ollama
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = []
        missing_texts: list[str] = []
        missing_hashes: list[str] = []

        for text in texts:
            text_hash = sha256_text(text)
            cached = self.db.get_embedding(text_hash, self.model)
            if cached is None:
                vectors.append(None)
                missing_texts.append(text)
                missing_hashes.append(text_hash)
            else:
                vectors.append(cached)

        if missing_texts:
            generated = self.ollama.embed(self.model, missing_texts)
            if len(generated) != len(missing_texts):
                raise RuntimeError("Ollama returned an unexpected number of embeddings.")
            generated_iter = iter(generated)
            for index, vector in enumerate(vectors):
                if vector is None:
                    new_vector = next(generated_iter)
                    vectors[index] = new_vector
            for text_hash, vector in zip(missing_hashes, generated):
                self.db.set_embedding(text_hash, self.model, vector)

        return [vector for vector in vectors if vector is not None]
