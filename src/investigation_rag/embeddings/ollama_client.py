from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str
    timeout: int = 120

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not model:
            raise OllamaError("OLLAMA_EMBED_MODEL is not configured.")
        if not texts:
            return []
        try:
            import requests
        except ImportError as exc:
            raise OllamaError("requests is required for Ollama integration.") from exc

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return [self._embed_legacy(requests, model, text) for text in texts]
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise OllamaError(f"Could not reach local Ollama embeddings endpoint: {exc}") from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise OllamaError("Ollama embeddings response did not include embeddings.")
        return embeddings

    def _embed_legacy(self, requests_module: Any, model: str, text: str) -> list[float]:
        response = requests_module.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise OllamaError("Ollama legacy embeddings response did not include embedding.")
        return embedding

    def generate(self, model: str, prompt: str) -> str:
        if not model:
            raise OllamaError("OLLAMA_LLM_MODEL is not configured.")
        try:
            import requests
        except ImportError as exc:
            raise OllamaError("requests is required for Ollama integration.") from exc

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise OllamaError(f"Could not reach local Ollama generation endpoint: {exc}") from exc
        return str(payload.get("response", "")).strip()
