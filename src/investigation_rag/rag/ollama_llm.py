from __future__ import annotations

from investigation_rag.embeddings.ollama_client import OllamaClient


class OllamaLLM:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def answer(self, prompt: str) -> str:
        return self.client.generate(self.model, prompt)
